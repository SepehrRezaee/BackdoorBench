import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, E, H/P, W/P
        x = x.flatten(2)  # B, E, N
        x = x.transpose(1, 2)  # B, N, E
        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, ff_size=2048, dropout=0.1):
        super().__init__()
        self.enc_layer = TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(self.enc_layer, num_layers=6)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_classes=10, img_size=224, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        self.transformer_block = TransformerBlock(emb_size=emb_size)
        self.head = nn.Linear(emb_size, num_classes)  # Classifier head

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_block(x)
        x = x.mean(dim=1)  # Global average pooling simulation
        x = self.head(x)
        return x


def test_vit():
    model = VisionTransformer(num_classes=10, img_size=32, patch_size=8)  # CIFAR-10 example
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.size())

# test_vit()
