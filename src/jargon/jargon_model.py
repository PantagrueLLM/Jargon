import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from fairseq.models.roberta import (
    RobertaModel as RobertModel, 
    RobertaEncoder as RobertaEncoderFS
)
from transformers.models.roberta.modeling_roberta import (
    RobertaEncoder,
    RobertaConfig,
    RobertaModel,
    RobertaLMHead,
    RobertaForMaskedLM,
    RobertaEmbeddings,
    RobertaForTokenClassification,
    RobertaForSequenceClassification
)
from transformers.modeling_outputs import (
    MaskedLMOutput,   
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

from .linformer import LinformerTransformerEncoderLayer
from .jargon_configuration import JargonConfig


class JargonModel(RobertaModel):
    config_class = JargonConfig
    def __init__(self, config, **kwargs):
        config_class = JargonConfig
        base_model_prefix = "jargon"
        
        super().__init__(config, **kwargs)
        self.embeddings = JargonEmbeddings(config)
        self.encoder = JargonEncoder(config)
    # Copied from modeling_roberta.py
    # Add transpose of embeddings as implemented in fairseq
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )


        embedding_output = embedding_output.transpose(0,1)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0].transpose(0,1)

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # Fairseq Linformer implementation works with transposed hidden states -> we transpose them back for HF implementation.
        if output_hidden_states:
            encoder_outputs.hidden_states = [h.transpose(0,1) for h in encoder_outputs.hidden_states]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class JargonForSequenceClassification(RobertaForSequenceClassification):

    config_class = JargonConfig
    
    def __init__(self, config,  **kwargs):
        base_model_prefix = "jargon"
        
        super().__init__(config, **kwargs)

        self.roberta = JargonModel(config, add_pooling_layer=False)
        self.sbo_head = self.build_sbo_head(config)

    def build_sbo_head(self, config):
        return SBOHead(
            config,
            embedding_weights=(
                self.roberta.embeddings.word_embeddings.weight
                if not config.untie_weights_roberta
                else None
            )
        )


class JargonForTokenClassification(RobertaForTokenClassification):

    config_class = JargonConfig
    
    def __init__(self, config,  **kwargs):
        base_model_prefix = "jargon"
        
        super().__init__(config, **kwargs)

        self.roberta = JargonModel(config, add_pooling_layer=False)
        self.sbo_head = self.build_sbo_head(config)

    def build_sbo_head(self, config):
        return SBOHead(
            config,
            embedding_weights=(
                self.roberta.embeddings.word_embeddings.weight
                if not config.untie_weights_roberta
                else None
            )
        )
        

class JargonForMaskedLM(RobertaForMaskedLM):

    config_class = JargonConfig
    
    def __init__(self, config,  **kwargs):
        base_model_prefix = "jargon"
        
        super().__init__(config, **kwargs)

        self.roberta = JargonModel(config, add_pooling_layer=False)
        self.sbo_head = self.build_sbo_head(config)

    def build_sbo_head(self, config):
        return SBOHead(
            config,
            embedding_weights=(
                self.roberta.embeddings.word_embeddings.weight
                if not config.untie_weights_roberta
                else None
            )
        )


class JargonEmbeddings(RobertaEmbeddings):

    def __init__(self, config, **kwargs):
        config_class = JargonConfig
        base_model_prefix = "jargon"
        super().__init__(config, **kwargs)
    
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)

        embeddings += position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class JargonEncoder(RobertaEncoder):

    def __init__(self, args):
        compress_layer = None
        if args.shared_layer_kv_compressed == 1 and compress_layer is None:
            compress_layer = nn.Linear(
                args.max_positions,
                args.max_positions // args.compressed
            )
            # intialize parameters for compressed layer
            nn.init.xavier_uniform_(compress_layer.weight, gain=1 / math.sqrt(2))
            if args.freeze_compress == 1:
                compress_layer.weight.requires_grad = False
            compress_layer = compress_layer

        super().__init__(args)
 
        self.layer = nn.ModuleList([LinformerTransformerEncoderLayer(args, compress_layer) for _ in range(args.num_layers)])
        self.compress_layer = compress_layer

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.embed_dim)
        else:
            self.layer_norm = None
        
        self.lm_head = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        x = super().forward(hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            head_mask=head_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            past_key_values=past_key_values,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)
        
 
        if self.layer_norm is not None:
            x.last_hidden_state = self.layer_norm(x.last_hidden_state)
        
        return x

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = LinformerTransformerEncoder(args)
        return encoder
        if args.use_linformer:
            encoder = LinformerTransformerEncoder(args, dictionary, embed_tokens)
        elif args.use_fft:
            encoder = FourierTransformerEncoder(args, dictionary, embed_tokens)
        else:
            encoder = TransformerEncoder(args, dictionary, embed_tokens)

        encoder.apply(init_bert_params)

        return encoder

    def output_layer(self, features, masked_tokens=None, pairs=None, **unused):
        lm_out = self.lm_head(features, masked_tokens)
        if pairs is not None:
            sbo_out = self.sbo_head(features, pairs)
            return lm_out, sbo_out
        else:
            return lm_out


class SBOLayer(nn.Module):

    def __init__(self, input_size, hidden_size, activation, export):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        self.activ = get_activation_fn(activation)
        self.norm = LayerNorm(hidden_size)

    def forward(self, x):
        return self.norm(self.activ(self.layer(x)))


class SBONetwork(nn.Module):

    def __init__(self, input_size, hidden_size, activation, export):
        super().__init__()
        self.layers = nn.ModuleList([
            self.build_sbo_layer(input_size, hidden_size, activation, export),
            self.build_sbo_layer(hidden_size, hidden_size, activation, export)
        ])
        self.layers = nn.Sequential(*self.layers)

    def build_sbo_layer(self, input_size, output_size, activation, export):
        return SBOLayer(input_size, output_size, activation, export)

    def forward(self, x):
        return self.layers(x)
 

class SBOHead(nn.Module):

    def __init__(self, args, embedding_weights, max_targets=10, position_embedding_size=200):
        super().__init__()

        self.position_embeddings = nn.Embedding(max_targets, position_embedding_size)

        export = getattr(args, "export", False)
        hidden_size = args.embed_dim
        input_size = hidden_size * 2 + position_embedding_size
        activation = getattr(args, "activation_fn", "relu") or "relu"

        self.mlp_layer_norm = self.build_sbo_network(input_size, hidden_size, activation, export)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            embedding_weights.size(1),
            embedding_weights.size(0),
            bias=False
        )
        if embedding_weights is not None:
            self.decoder.weight = embedding_weights

        self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))
        self.max_targets = max_targets

    def build_sbo_network(self, input_size, hidden_size, activation, export):
        return SBONetwork(input_size, hidden_size, activation, export)

    def forward(self, hidden_states, pairs):
        bs, num_pairs, _ = pairs.size()
        bs, seq_len, dim = hidden_states.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:,:, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim))
        # pair states: bs * num_pairs, max_targets, dim
        left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        
        right_hidden = torch.gather(hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim))
        # bs * num_pairs, max_targets, dim
        right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)

        # (max_targets, dim)
        position_embeddings = self.position_embeddings.weight
       
        z = torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)), -1)
        
        hidden_states = self.mlp_layer_norm(torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)), -1))
        # target scores : bs * num_pairs, max_targets, vocab_size
        target_scores = self.decoder(hidden_states) + self.bias
        return target_scores


def get_activation_fn(activation):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return F.relu_squared
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return F.gelu_accurate
    elif activation == "gelu_accurate":
        return F.gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
