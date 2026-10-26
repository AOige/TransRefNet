import math
import torch
import torch.nn as nn
from resnet import resnet50


# ──────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────────────────────────────────────

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1,
                 bias=False, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                      padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.network = nn.Sequential(
            Conv2D(in_c, out_c),
            Conv2D(out_c, out_c, kernel_size=1, padding=0, act=False)
        )
        self.shortcut = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_init):
        x = self.network(x_init)
        s = self.shortcut(x_init)
        return self.relu(x + s)


class encoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.c1 = Conv2D(64,   ch)
        self.c2 = Conv2D(256,  ch)
        self.c3 = Conv2D(512,  ch)
        self.c4 = Conv2D(1024, ch)

    def forward(self, x):
        x1 = self.layer0(x)        # [-1, 64,   h/2,  w/2]
        x2 = self.layer1(x1)       # [-1, 256,  h/4,  w/4]
        x3 = self.layer2(x2)       # [-1, 512,  h/8,  w/8]
        x4 = self.layer3(x3)       # [-1, 1024, h/16, w/16]

        return self.c1(x1), self.c2(x2), self.c3(x3), self.c4(x4)


class context_feature_refinement(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.d1 = Conv2D(in_c, out_c, kernel_size=1,  padding=0)
        self.d2 = Conv2D(in_c, out_c, kernel_size=3,  padding=1)
        self.d3 = Conv2D(in_c, out_c, kernel_size=7,  padding=3)
        self.d4 = Conv2D(in_c, out_c, kernel_size=11, padding=5)
        self.c1 = Conv2D(out_c * 4, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        x = torch.cat([self.d1(x), self.d2(x), self.d3(x), self.d4(x)], dim=1)
        return self.c1(x)


# ──────────────────────────────────────────────────────────────────────────────
# Original ConvSegNet (unchanged — keep for baseline)
# ──────────────────────────────────────────────────────────────────────────────

class ConvSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder(64)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = context_feature_refinement(128, 64)
        self.c2 = context_feature_refinement(128, 64)
        self.c3 = context_feature_refinement(128, 64)
        self.s0 = Conv2D(3, 64)
        self.c4 = context_feature_refinement(128, 64)
        self.output = Conv2D(64, 1, kernel_size=1, padding=0)

    def forward(self, image):
        s1, s2, s3, s4 = self.encoder(image)

        x = self.up(s4)
        x = self.c1(torch.cat([x, s3], dim=1))

        x = self.up(x)
        x = self.c2(torch.cat([x, s2], dim=1))

        x = self.up(x)
        x = self.c3(torch.cat([x, s1], dim=1))

        x = self.up(x)
        s0 = self.s0(image)
        x = self.c4(torch.cat([x, s0], dim=1))

        return self.output(x)


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Transformer Bridge Module
# ──────────────────────────────────────────────────────────────────────────────

class LearnedPositionalEmbedding2D(nn.Module):
    """
    Learnable 2-D positional embeddings factorised along H and W.
    For a 16×16 feature map (H_max=W_max=32 to allow flexible sizes),
    the total embedding is emb_h[h] + emb_w[w].  Factorised so the
    embedding table stays small and generalises to slightly different
    spatial sizes at test time.
    """
    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 32):
        super().__init__()
        self.emb_h = nn.Embedding(max_h, d_model)
        self.emb_w = nn.Embedding(max_w, d_model)
        nn.init.trunc_normal_(self.emb_h.weight, std=0.02)
        nn.init.trunc_normal_(self.emb_w.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H*W, C]  (caller provides H, W separately)"""
        raise NotImplementedError

    def get_encoding(self, H: int, W: int, device) -> torch.Tensor:
        """Returns [1, H*W, d_model] positional encoding."""
        rows = torch.arange(H, device=device)   # [H]
        cols = torch.arange(W, device=device)   # [W]
        emb_r = self.emb_h(rows)                # [H, d]
        emb_c = self.emb_w(cols)                # [W, d]
        # broadcast: [H, 1, d] + [1, W, d] → [H, W, d] → [H*W, d]
        pe = (emb_r.unsqueeze(1) + emb_c.unsqueeze(0)).reshape(H * W, -1)
        return pe.unsqueeze(0)                  # [1, H*W, d]


class TransformerBridge(nn.Module):
    """
    Lightweight Transformer bottleneck applied to the deepest encoder feature.

    For ConvSegNet with 256×256 input and ch=64:
        c4 is [B, 64, 16, 16] → 256 tokens of dim 64.
    Two Pre-LN Transformer encoder layers add ~2.1 M params.

    Args:
        in_c       : channel dimension of the incoming feature (default 64).
        num_heads  : MHSA heads.  Must divide in_c evenly.
        num_layers : depth of the Transformer stack.
        mlp_ratio  : FFN hidden dim = in_c * mlp_ratio.
        dropout    : applied inside the Transformer.
    """
    def __init__(self, in_c: int = 64, num_heads: int = 8,
                 num_layers: int = 2, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()

        assert in_c % num_heads == 0, \
            f"in_c ({in_c}) must be divisible by num_heads ({num_heads})"

        self.pos_embed = LearnedPositionalEmbedding2D(in_c)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_c,
            nhead=num_heads,
            dim_feedforward=int(in_c * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN: more stable training
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(in_c)
        )

        # Lightweight projection back to spatial feature
        self.out_proj = nn.Sequential(
            nn.Linear(in_c, in_c),
            nn.LayerNorm(in_c)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, H, W]
        returns: same shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. Flatten spatial dims → token sequence
        tokens = x.flatten(2).transpose(1, 2)          # [B, H*W, C]

        # 2. Add 2-D positional embeddings
        pe = self.pos_embed.get_encoding(H, W, x.device)  # [1, H*W, C]
        tokens = tokens + pe

        # 3. Transformer encoding
        tokens = self.transformer(tokens)               # [B, H*W, C]

        # 4. Output projection
        tokens = self.out_proj(tokens)                  # [B, H*W, C]

        # 5. Residual: add back original flattened feature
        tokens = tokens + x.flatten(2).transpose(1, 2)

        # 6. Reshape to spatial
        out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# TransRefNet: Transformer-augmented Refinement Network for biomedical
#              image segmentation
# ──────────────────────────────────────────────────────────────────────────────

class TransRefNet(nn.Module):
    """
    TransRefNet — a new biomedical image segmentation architecture that fuses
    CNN-based local feature refinement with Transformer-based global context
    modelling.

    Core idea:
      A ResNet50 encoder extracts multi-scale features.  Before decoding, the
      deepest feature map (c4, h/16) is passed through a lightweight Transformer
      Bridge that applies global self-attention over 256 spatial tokens, capturing
      long-range anatomical dependencies that the convolutional CFR modules cannot
      model.  The enriched bottleneck features are then decoded through the same
      Context Feature Refinement (CFR) skip-connection structure used in the
      baseline ConvSegNet.

    Architectural change vs. ConvSegNet:
        Encoder → c4 → TransformerBridge → Decoder
                                         ↑
                   (c1, c2, c3 skip connections unchanged)

    Param budget: +~2.1 M over ConvSegNet (num_heads=8, num_layers=2).
    """
    def __init__(self, ch: int = 64, tb_num_heads: int = 8,
                 tb_num_layers: int = 2, tb_mlp_ratio: float = 4.0,
                 tb_dropout: float = 0.1):
        super().__init__()

        # ── Encoder (identical to ConvSegNet) ─────────────────────────────
        self.encoder = encoder(ch)

        # ── Transformer Bridge ────────────────────────────────────────────
        self.transformer_bridge = TransformerBridge(
            in_c=ch,
            num_heads=tb_num_heads,
            num_layers=tb_num_layers,
            mlp_ratio=tb_mlp_ratio,
            dropout=tb_dropout
        )

        # ── Decoder (identical to ConvSegNet) ─────────────────────────────
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = context_feature_refinement(ch * 2, ch)
        self.c2 = context_feature_refinement(ch * 2, ch)
        self.c3 = context_feature_refinement(ch * 2, ch)
        self.s0 = Conv2D(3, ch)
        self.c4 = context_feature_refinement(ch * 2, ch)

        self.output = Conv2D(ch, 1, kernel_size=1, padding=0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # 1. Encode
        s1, s2, s3, s4 = self.encoder(image)   # all [B, ch, h/2~h/16, ...]

        # 2. Transformer Bridge on deepest feature
        s4 = self.transformer_bridge(s4)        # [B, ch, h/16, w/16]

        # 3. Decode (same as ConvSegNet)
        x = self.up(s4)
        x = self.c1(torch.cat([x, s3], dim=1))

        x = self.up(x)
        x = self.c2(torch.cat([x, s2], dim=1))

        x = self.up(x)
        x = self.c3(torch.cat([x, s1], dim=1))

        x = self.up(x)
        s0 = self.s0(image)
        x = self.c4(torch.cat([x, s0], dim=1))

        return self.output(x)


# ──────────────────────────────────────────────────────────────────────────────
# Complexity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    print("=" * 55)
    print("ConvSegNet (baseline)")
    print("=" * 55)
    model_base = ConvSegNet()
    flops, params = get_model_complexity_info(
        model_base, input_res=(3, 256, 256),
        as_strings=True, print_per_layer_stat=False
    )
    print(f"  FLOPs  : {flops}")
    print(f"  Params : {params}")

    print()
    print("=" * 55)
    print("TransRefNet (proposed)")
    print("=" * 55)
    model_new = TransRefNet()
    flops2, params2 = get_model_complexity_info(
        model_new, input_res=(3, 256, 256),
        as_strings=True, print_per_layer_stat=False
    )
    print(f"  FLOPs  : {flops2}")
    print(f"  Params : {params2}")

    # Quick forward pass sanity check
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y1 = model_base(x)
        y2 = model_new(x)
    print()
    print(f"ConvSegNet output:  {tuple(y1.shape)}")
    print(f"TransRefNet output: {tuple(y2.shape)}")
    assert y1.shape == y2.shape, "Output shape mismatch!"
    print("Shape check: PASSED")
