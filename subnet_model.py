import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask
from typing import Optional, Tuple, List

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

class SubnetLLM(BaseModel):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        embedding_layers: int,
        coherence_layers: int,
        compensation_layers: int,
        concatenation_layers: int,
        adaptation_layers: int,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        freeze_base_model: bool = True,
    ):
        super().__init__()

        self.device = device
        self._freeze_base_model = freeze_base_model

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )

        total_layers = len(base_model.model.layers)
        reasoning_layers = total_layers - embedding_layers - coherence_layers

        assert embedding_layers < total_layers
        assert reasoning_layers > 0

        self.rotary_emb = base_model.model.rotary_emb.to(device)
        self.embed_tokens = base_model.model.embed_tokens.to(device)

        self.embedding_subnet = TransformerSubnet(
            base_model.model.layers[:embedding_layers]
        ).to(device)

        self.reasoning_subnet = TransformerSubnet(
            base_model.model.layers[embedding_layers:embedding_layers + reasoning_layers]
        ).to(device)

        # Get the decoder layer class and dtype from the base model
        layer_class = type(base_model.model.layers[0])
        model_dtype = next(base_model.model.layers[0].parameters()).dtype

        self.adaptation_subnet = MLPSubnet(
            hidden_size=base_model.config.hidden_size,
            num_layers=adaptation_layers,
            dtype=model_dtype,
            device=device
        ).to(dtype=model_dtype, device=device)

        self.compensation_subnet = CompensationSubnet(
            layer_class=layer_class,
            config=base_model.config,
            num_layers=compensation_layers
        ).to(dtype=model_dtype, device=device)

        self.concatenation_subnet = MLPSubnet(
            hidden_size=base_model.config.hidden_size,
            num_layers=concatenation_layers,
            dtype=model_dtype,
            device=device
        ).to(dtype=model_dtype, device=device)

        self.coherence_subnet = CoherenceSubnet(
            base_model.model.layers[embedding_layers + reasoning_layers:],
            base_model.model.norm,
            base_model.lm_head
        ).to(device)

        if freeze_base_model:
            self._apply_freeze()

        self.config = base_model.config
        del base_model

    def _apply_freeze(self) -> None:
        """Freeze base model parts (embedding, reasoning, coherence subnets)."""
        for module in (self.rotary_emb, self.embed_tokens,
                       self.embedding_subnet, self.reasoning_subnet,
                       self.coherence_subnet):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        cached_reasoning_outputs: Optional[List[Optional[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = False,
        prompt_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with support for both autoregressive inference and teacher forcing training.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            cached_reasoning_outputs: Cached reasoning outputs for autoregressive generation
            attention_mask: Attention mask [batch_size, seq_length]
            use_teacher_forcing: If True, use position-based masking for parallel training
            prompt_length: For teacher forcing, specifies where prompt ends (positions < prompt_length use reasoning,
                          positions >= prompt_length use adaptation+compensation). If None, defaults to 1.

        Returns:
            logits: Output logits [batch_size, seq_length, vocab_size]
            cached_reasoning_outputs: Updated cache (None if use_teacher_forcing=True)
        """
        # Initialize cache for backwards compatibility
        if cached_reasoning_outputs is None:
            batch_size = input_ids.shape[0]
            cached_reasoning_outputs = [None] * batch_size

        if use_teacher_forcing:
            return self._forward_teacher_forcing(input_ids, attention_mask, prompt_length)
        else:
            return self._forward_autoregressive(input_ids, cached_reasoning_outputs, attention_mask)

    def _forward_teacher_forcing(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Teacher forcing forward pass - processes entire sequence in parallel.
        Positions [0, prompt_length) use reasoning subnet (simulating initial cache setup).
        Positions [prompt_length, seq_length) use adaptation+compensation with cache from prompt.
        """
        batch_size, seq_length = input_ids.shape

        # Default prompt_length to 1 if not specified (backward compatibility)
        if prompt_length is None:
            prompt_length = 1

        # Step 1: Embedding and positional encoding
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cache_position = torch.arange(seq_length, device=input_ids.device)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        embedding_output = self.embedding_subnet(hidden_states, position_embeddings, causal_mask)

        # Initialize logits tensor
        all_logits = []

        # For prompt positions: compute reasoning output and logits (matches AR with no cache)
        # This is the ONLY time we compute reasoning outputs - they get cached for reuse
        prompt_embedding = embedding_output[:, :prompt_length, :]
        prompt_position_embeddings = tuple(pe[:, :prompt_length, ...] for pe in position_embeddings)
        prompt_cache_position = torch.arange(prompt_length, device=input_ids.device)
        prompt_position_ids = torch.arange(prompt_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        prompt_causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=prompt_embedding,
            attention_mask=attention_mask[:, :prompt_length] if attention_mask is not None else None,
            cache_position=prompt_cache_position,
            past_key_values=None,
            position_ids=prompt_position_ids,
        )

        # Compute reasoning outputs ONLY for the prompt (positions 0 to prompt_length-1)
        # This creates the "cache" that will be reused for all generated positions
        cached_reasoning = self.reasoning_subnet(prompt_embedding, prompt_position_embeddings, prompt_causal_mask)

        # Compute prompt logits
        prompt_logits = self.coherence_subnet(
            cached_reasoning,
            prompt_position_embeddings,
            prompt_causal_mask
        )
        all_logits.append(prompt_logits)

        # For generated positions: use adaptation+compensation+concatenation (matches AR with cache)
        # The cached_reasoning from the prompt is reused for ALL generated positions (never recomputed)
        # Process all generated positions in parallel using masking
        if seq_length > prompt_length:
            # Pad cached reasoning to full sequence length
            pad_size = seq_length - cached_reasoning.shape[1]
            padding = torch.zeros(
                (batch_size, pad_size, cached_reasoning.shape[2]),
                device=cached_reasoning.device,
                dtype=cached_reasoning.dtype
            )
            cached_reasoning_padded = torch.cat([cached_reasoning, padding], dim=1)

            # Apply adaptation subnet to the full padded cache
            adaptation_output = self.adaptation_subnet(cached_reasoning_padded)

            # Apply compensation subnet to the full embedding sequence
            compensation_output = self.compensation_subnet(
                embedding_output,
                position_embeddings,
                causal_mask
            )

            # Combine and pass through concatenation subnet
            concatenation_output = self.concatenation_subnet(adaptation_output + compensation_output)

            # Pass through coherence subnet to get logits
            # Use a special mask that makes each generated position behave as if sequence ends there
            generated_logits = self.coherence_subnet(
                concatenation_output,
                position_embeddings,
                causal_mask
            )

            # Extract only the logits for generated positions
            all_logits.append(generated_logits[:, prompt_length:, :])

        # Concatenate all logits
        logits = torch.cat(all_logits, dim=1)

        return logits, None

    def _forward_autoregressive(
        self,
        input_ids: torch.Tensor,
        cached_reasoning_outputs: List[Optional[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Original autoregressive forward pass for inference.
        """
        batch_size, seq_length = input_ids.shape

        # Step 1: Embedding and positional encoding
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cache_position = torch.arange(seq_length, device=input_ids.device)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        embedding_output = self.embedding_subnet(hidden_states, position_embeddings, causal_mask)
        coherence_subnet_inputs = torch.zeros((batch_size, seq_length, embedding_output.size(-1)), device=input_ids.device, dtype=hidden_states.dtype)

        first_token_sequence_ids = [i for i in range(len(cached_reasoning_outputs)) if cached_reasoning_outputs[i] is None]
        non_first_token_sequence_ids = [i for i in range(len(cached_reasoning_outputs)) if cached_reasoning_outputs[i] is not None]

        if len(first_token_sequence_ids) > 0:
            first_token_embedding_outputs = embedding_output[first_token_sequence_ids]
            first_token_position_embeddings = tuple(pe[first_token_sequence_ids] for pe in position_embeddings)
            first_token_causal_mask = causal_mask[first_token_sequence_ids] if causal_mask is not None else None

            reasoning_outputs = self.reasoning_subnet(first_token_embedding_outputs, first_token_position_embeddings, first_token_causal_mask)
            coherence_subnet_inputs[first_token_sequence_ids] = reasoning_outputs

            for i in range(len(first_token_sequence_ids)):
                # Detach to avoid backpropagating through cached values in autoregressive training
                cached_reasoning_outputs[i] = reasoning_outputs[i:i+1, :, :].detach()

        if len(non_first_token_sequence_ids) > 0:
            non_first_token_embedding_outputs = embedding_output[non_first_token_sequence_ids]
            non_first_token_position_embeddings = tuple(pe[non_first_token_sequence_ids] for pe in position_embeddings)
            non_first_token_causal_mask = causal_mask[non_first_token_sequence_ids] if causal_mask is not None else None

            # Find the maximum cached length across all non-first-token sequences
            max_cached_length = max(
                cached_reasoning_outputs[i].size(1)
                for i in non_first_token_sequence_ids
            )

            # Pad to the maximum of (max_cached_length, seq_length)
            target_length = max(max_cached_length, seq_length)

            # Pad each cached output to target_length before concatenation
            padded_caches = []
            for i in non_first_token_sequence_ids:
                cache = cached_reasoning_outputs[i]
                if cache.size(1) < target_length:
                    pad_size = target_length - cache.size(1)
                    padding = torch.zeros(
                        (1, pad_size, cache.size(2)),
                        device=cache.device,
                        dtype=cache.dtype
                    )
                    padded_cache = torch.cat([cache, padding], dim=1)
                else:
                    padded_cache = cache
                padded_caches.append(padded_cache)

            non_first_token_reasoning_outputs = torch.cat(padded_caches, dim=0)

            adaptation_outputs = self.adaptation_subnet(non_first_token_reasoning_outputs)

            # Trim adaptation_outputs to match seq_length (the compensation_outputs length)
            adaptation_outputs = adaptation_outputs[:, :seq_length, :]

            compensation_outputs = self.compensation_subnet(non_first_token_embedding_outputs, non_first_token_position_embeddings, non_first_token_causal_mask)
            concatenation_outputs = self.concatenation_subnet(adaptation_outputs + compensation_outputs)
            coherence_subnet_inputs[non_first_token_sequence_ids] = concatenation_outputs

        logits = self.coherence_subnet(coherence_subnet_inputs, position_embeddings, causal_mask)

        return logits, cached_reasoning_outputs # pyright: ignore[reportReturnType]

    def train(self, mode: bool = True) -> nn.Module:
        self.training = mode
        if self._freeze_base_model:
            # Keep frozen parts in eval; only switch trainable subnets
            self.adaptation_subnet.train(mode)
            self.compensation_subnet.train(mode)
            self.concatenation_subnet.train(mode)
        else:
            # Train everything
            super().train(mode)
        return self

class TransformerSubnet(BaseModel):
    def __init__(
        self,
        layers: nn.ModuleList,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = hidden_states

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_embeddings=position_embeddings)

        return x

class CompensationSubnet(BaseModel):
    def __init__(
        self,
        layer_class: type,
        config: PretrainedConfig,
        num_layers: int
    ):
        super().__init__()
        self.layer_class = layer_class
        # Create new transformer layers with random initialization
        self.layers = nn.ModuleList([
            layer_class(config, layer_idx=i)
            for i in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = hidden_states

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_embeddings=position_embeddings)

        return x

    def get_layer_type(self) -> str:
        return self.layer_class.__name__

    def num_layers(self) -> int:
        return len(self.layers)

class MLPSubnet(BaseModel):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dtype: torch.dtype,
        device: str
    ):
        super().__init__()

        self._layer_class = nn.Linear

        self.dense_layers = nn.ModuleList()

        for _ in range(num_layers):
            linear = self._layer_class(hidden_size, hidden_size, device=device, dtype=dtype)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.dense_layers.append(linear)

        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states

        for layer in self.dense_layers:
            x = self.activation(layer(x))

        return x

    def get_layer_type(self) -> str:
        return self._layer_class.__name__

    def num_layers(self) -> int:
        return len(self.dense_layers)

class CoherenceSubnet(BaseModel):
    def __init__(
        self,
        layers: nn.ModuleList,
        norm: nn.Module,
        lm_head: nn.Module
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = hidden_states

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_embeddings=position_embeddings)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits
