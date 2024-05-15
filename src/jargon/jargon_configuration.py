
from transformers.models.roberta.modeling_roberta import RobertaConfig


class JargonConfig(RobertaConfig):
    model_type = "jargon"

    def __init__(
        self,
        compress_layer= 1,
        shared_layer_kv_compressed=1,
        shared_kv_compressed=0,
        max_positions=512,
        max_position_embeddings=512,
        compressed=4,
        vocab_size=30522,
        freeze_compress=0,
        embed_dim=768,
        num_heads=16,
        dim_feedforward=4096,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-05,
        self_attention=True,
        encoder_decoder_attention=False,
        bias=True,
        q_noise=0,
        qn_block_size=8,
        add_bias_kv=False,
        add_zero_attn=False,
        num_layers=12,
        untie_weights_roberta=False,
        layernorm_embedding=False,
        encoder_normalize_before=False,
        encoder_embed_dim=768,
        encoder_attention_heads=12,
        quant_noise_pq=0.0,
        quant_noise_pq_block_size=8,
        quant_noise_scalar=0,
        encoder_ffn_embed_dim=4096,
        add_pooling_layer=False,
        intermediate_size=4096,
        intermediate_act_fn="relu",
        hidden_act="relu",
        output_hidden_states=False,
        position_embedding_type="learned",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.add_pooling_layer = add_pooling_layer
        self.compress_layer = compress_layer
        self.shared_layer_kv_compressed = shared_layer_kv_compressed
        self.shared_kv_compressed = shared_kv_compressed
        self.max_positions = max_positions
        self.max_position_embeddings = max_position_embeddings
        self.compressed = compressed
        self.freeze_compress = freeze_compress
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward=dim_feedforward
        self.dropout = dropout
        self.activation= activation 
        self.layer_norm_eps = layer_norm_eps
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.bias = bias
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.num_layers = num_layers
        self.untie_weights_roberta = untie_weights_roberta
        self.layernorm_embedding=layernorm_embedding
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_attention_heads=encoder_attention_heads
        self.quant_noise_pq = quant_noise_pq
        self.quant_noise_pq_block_size=quant_noise_pq_block_size
        self.quant_noise_scalar=quant_noise_scalar
        self.encoder_normalize_before=encoder_normalize_before
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.output_hidden_states = output_hidden_states
        self.hidden_act = hidden_act
        self.position_embedding_type = position_embedding_type
        self.encoder_normalize_before = encoder_normalize_before
