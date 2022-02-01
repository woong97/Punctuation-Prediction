import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATION_FUNCTIONS = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "elu": nn.ELU()}


# Reference : F.multi_head_attention_forward (torch library)
class SelfAttention(nn.Module):
    def __init__(self, model_cfg):
        super(SelfAttention, self).__init__()
        self.model_cfg = model_cfg

        self.hidden_size = model_cfg['hidden_size']
        self.q = nn.Linear(self.hidden_size, self.hidden_size)
        self.k = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = nn.Linear(self.hidden_size, self.hidden_size)

        self.num_heads = model_cfg['num_heads']
        self.head_dim = self.hidden_size // self.num_heads

        self.scale = self.head_dim ** 0.5

        self.dropout = nn.Dropout(p=model_cfg['dropout'])
        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def reshape_for_multihead(self, x):
        return x.contiguous().view(x.size(0), x.size(1) * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(self, query, key, value, attn_padding_mask):
        src_len, bsz, hidden_dim = query.size()

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        query = self.reshape_for_multihead(query)
        key = self.reshape_for_multihead(key)
        value = self.reshape_for_multihead(value)

        tgt_len = key.size(1)

        assert src_len == tgt_len, "For self-attention, src_len equals tgt_len"

        attn_score = torch.bmm(query, key.transpose(1, 2)) / self.scale

        if attn_padding_mask is not None:
            attn_score = attn_score.view(bsz, self.num_heads, tgt_len, src_len)
            attn_score = attn_score.masked_fill(
                attn_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_score = attn_score.view(bsz * self.num_heads, tgt_len, src_len)

        attn_score = F.softmax(
            attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        # attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

        attn_out = torch.bmm(attn_score, value)
        assert list(attn_out.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_out = attn_out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.hidden_size)
        attn_out = self.out(attn_out)
        return attn_out


class EncoderLayer(nn.Module):
    def __init__(self, model_cfg):
        super(EncoderLayer, self).__init__()
        self.model_cfg = model_cfg
        self.layer_dropout = nn.Dropout(p=model_cfg['dropout'])

        # attn layer
        self.self_attn = SelfAttention(model_cfg=model_cfg)
        self.attn_layer_norm = nn.LayerNorm(model_cfg['hidden_size'], eps=model_cfg['attn_layer_norm_eps'])

        # feed forward layer
        self.fc1 = nn.Linear(model_cfg['hidden_size'], model_cfg['ff_intermediate_size'])
        self.fc2 = nn.Linear(model_cfg['ff_intermediate_size'], model_cfg['hidden_size'])
        self.activation = ACTIVATION_FUNCTIONS[model_cfg['activation_fn']]
        self.ff_layer_norm = nn.LayerNorm(model_cfg['hidden_size'], eps=model_cfg['ff_layer_norm_eps'])

    def forward(self, hidden_states, attn_padding_mask):
        residual = hidden_states

        # attn layer
        hidden_states = self.self_attn(
                            query=hidden_states,
                            key=hidden_states,
                            value=hidden_states,
                            attn_padding_mask=attn_padding_mask
                        )
        hidden_states += residual
        hidden_states = self.layer_dropout(hidden_states)
        hidden_states = self.attn_layer_norm(hidden_states)

        residual = hidden_states

        # feed forward layer
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.layer_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)

        hidden_states += residual
        hidden_states = self.layer_dropout(hidden_states)
        hidden_states = self.ff_layer_norm(hidden_states)
        return hidden_states


class TransformerEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(TransformerEncoder, self).__init__()
        self.model_cfg = model_cfg
        self.layers = nn.Sequential(*[EncoderLayer(model_cfg) for _ in range(model_cfg['num_layers'])])

        self.use_linear_layer = False
        if model_cfg['embedding_dim'] != model_cfg['hidden_size']:
            self.use_linear_layer = True
            self.linear = nn.Linear(model_cfg['embedding_dim'], model_cfg['hidden_size'])
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(p=model_cfg['dropout'])

    def forward(self, hidden_states, attn_padding_mask):
        if self.use_linear_layer:
            hidden_states = self.linear(hidden_states)
            hidden_states = self.activation(hidden_states)
            hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states.transpose(0, 1)
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, attn_padding_mask=attn_padding_mask)
        return hidden_states