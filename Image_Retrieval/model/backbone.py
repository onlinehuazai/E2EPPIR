import torch
import torch.nn as nn
from .Transformer import TransformerModel
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim1,
        img_dim2,
        patch_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=6,
        num_layers=6,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        # assert img_dim % patch_dim == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim1 // patch_dim) *(img_dim2 // patch_dim))
        self.seq_length = self.num_patches + 1
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
                            embedding_dim,
                            num_layers,
                            num_heads,
                            hidden_dim,
                            self.dropout_rate,
                            self.attn_dropout_rate,
                        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        if self.conv_patch_representation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=(self.patch_dim, self.patch_dim),
                stride=(self.patch_dim, self.patch_dim),
                padding=self._get_padding(
                    'VALID', (self.patch_dim, self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()

    def forward(self, x):
        n, c, h, w = x.shape
        if self.conv_patch_representation:
            # combine embedding w/ conv patch distribution
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim)
                .unfold(3, self.patch_dim, self.patch_dim)
                .contiguous()
            )
            x = x.view(n, c, -1, self.patch_dim ** 2)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        cls_tokens = self.cls_token.expand(n, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)
        # apply transformer
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = self.to_cls_token(x[:, 0])
        return x

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

