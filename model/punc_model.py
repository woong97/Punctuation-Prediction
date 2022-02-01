import torch
import torch.nn as nn
import numpy as np
from model.transformer import TransformerEncoder, ACTIVATION_FUNCTIONS


# Reference : Huggingface Transformers
def create_sinusoidal_embeddings(max_position_len, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(max_position_len)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()


class PunctuationModel(nn.Module):

    def __init__(
        self,
        model_cfg,
        vocab_size,
        num_labels,
        pad_idx
    ):
        super(PunctuationModel, self).__init__()
        self.model_cfg = model_cfg
        self.pad_idx = pad_idx

        self.token_embeddings = nn.Embedding(
                                    num_embeddings=vocab_size,
                                    embedding_dim=model_cfg['embedding_dim'],
                                    padding_idx=pad_idx
                                    )

        # sinusoidal positional embeddings reference : Huggingface Transformers
        self.positional_embeddings = nn.Embedding(
                                        num_embeddings=model_cfg['max_position_len'],
                                        embedding_dim=model_cfg['embedding_dim']
                                        )
        create_sinusoidal_embeddings(
            max_position_len=model_cfg['max_position_len'],
            dim=model_cfg['embedding_dim'],
            out=self.positional_embeddings.weight
        )

        self.embedding_dropout = nn.Dropout(p=model_cfg['dropout'])

        self.encoder = TransformerEncoder(model_cfg=model_cfg)

        # Final out layer
        self.fc = nn.Linear(model_cfg['hidden_size'], model_cfg['out_middle_dim'])
        self.activation = ACTIVATION_FUNCTIONS[model_cfg['activation_fn']]
        self.dropout = nn.Dropout(p=model_cfg['dropout'])
        self.layernorm = nn.LayerNorm(model_cfg['out_middle_dim'], eps=model_cfg['out_layer_norm_eps'])

        self.out = nn.Linear(model_cfg['out_middle_dim'], num_labels)

    def forward(self, input_ids):
        token_embeddings = self.token_embeddings(input_ids)

        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.repeat(batch_size).view(batch_size, -1)
        position_embeddings = self.positional_embeddings(position_ids)

        x = token_embeddings + position_embeddings
        x = self.embedding_dropout(x)

        attn_padding_mask = (input_ids == self.pad_idx)
        x = self.encoder(hidden_states=x, attn_padding_mask=attn_padding_mask)

        x = x.transpose(0, 1)
        # out layer
        x = self.activation(self.fc(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
