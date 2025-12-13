import torch
from typing import Optional, List, Tuple, Union
from transformers. models.llama. modeling_llama import (
    LlamaModel,
    LlamaConfig,
)
from transformers. modeling_outputs import BaseModelOutputWithPast
from transformers. cache_utils import DynamicCache
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlamaModelLBW(LlamaModel):
    """
    Llama Model adapted for 'Look Both Ways' (LBW).  
    Inherits from LlamaModel and overrides attention mask generation. 
    Implements the MASK0 strategy: Bidirectional attention where the first token (sink)
    is masked from being attended to by other tokens.
    
    Only the last 4 layers are converted to bidirectional MASK0,
    while earlier layers remain causal. 
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        # LBW Configuration
        self.lbw_mask_type = "MASK0"
        
        # Number of layers to convert to bidirectional (from the end)
        # Only last 4 layers are bidirectional
        total_layers = config.num_hidden_layers
        self.num_bidirectional_layers = 4
        self.bidirectional_start_idx = total_layers - self.num_bidirectional_layers
        
        # Set is_causal flag for each layer
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'self_attn'):
                if idx >= self.bidirectional_start_idx:
                    layer.self_attn.is_causal = False  # Bidirectional
                else:
                    layer.self_attn.is_causal = True   # Keep causal
        
        print(f"LBW Configuration: {self.num_bidirectional_layers} bidirectional layers "
              f"(layers {self.bidirectional_start_idx} to {total_layers - 1}), "
              f"{self.bidirectional_start_idx} causal layers (layers 0 to {self.bidirectional_start_idx - 1})")

    def _create_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Creates a standard causal (autoregressive) attention mask.
        Tokens can only attend to previous tokens and themselves.
        
        Returns:
            Attention mask of shape [batch_size, 1, seq_length, seq_length]
        """
        # Create causal mask: upper triangle is -inf
        causal_mask = torch.full(
            (batch_size, 1, seq_length, seq_length),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype
        )
        # Fill lower triangle and diagonal with 0 (can attend)
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device, dtype=torch.bool), diagonal=1)
        causal_mask.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), 0.0)

        # Apply padding mask if provided
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].to(dtype)
            inverted_mask = (1.0 - padding_mask) * torch.finfo(dtype).min
            causal_mask = causal_mask + inverted_mask

        return causal_mask

    def _create_lbw_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Creates the LBW attention mask with MASK0 strategy.
        - Bidirectional attention (not causal)
        - First token (position 0) is masked from being attended to by tokens at position > 0
        
        Returns:
            Attention mask of shape [batch_size, 1, seq_length, seq_length]
        """
        # Start with bidirectional mask (all zeros = all positions can attend)
        attn_mask = torch.zeros((batch_size, 1, seq_length, seq_length), device=device, dtype=dtype)

        # Apply padding mask if provided
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].to(dtype)
            inverted_mask = (1.0 - padding_mask) * torch.finfo(dtype).min
            attn_mask = attn_mask + inverted_mask

        # Apply MASK0 pattern: tokens at position > 0 cannot attend to token at position 0
        if self.lbw_mask_type == "MASK0" and seq_length > 1:
            attn_mask[:, :, 1:, 0] = torch.finfo(dtype).min

        return attn_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Handle default values
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Validate inputs
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Handle cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Set up cache_position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=device
            )

        # Set up position_ids
        if position_ids is None:
            position_ids = cache_position. unsqueeze(0)

        # Create BOTH masks (we'll use different ones for different layers)
        causal_mask = self._create_causal_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_length=seq_length,
            dtype=dtype,
            device=device,
        )
        
        lbw_mask = self._create_lbw_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_length=seq_length,
            dtype=dtype,
            device=device,
        )

        hidden_states = inputs_embeds

        # Compute rotary position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Collect outputs if requested
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Forward through decoder layers
        for idx, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Choose mask based on layer index
            # Layers 0 to (bidirectional_start_idx - 1): Causal
            # Layers bidirectional_start_idx to end: Bidirectional MASK0
            if idx >= self.bidirectional_start_idx:
                layer_mask = lbw_mask  # Bidirectional MASK0 for last 4 layers
            else:
                layer_mask = causal_mask  # Causal for earlier layers

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values if use_cache else None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if output_attentions and isinstance(layer_outputs, tuple) and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return outputs
        if not return_dict:
            outputs = (hidden_states, past_key_values if use_cache else None)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)
            return tuple(v for v in outputs if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
