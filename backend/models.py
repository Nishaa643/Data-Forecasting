"""Model classes extracted from the forecasting notebook."""

import math
import numpy as np
import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    """Decomposes time-series into trend and seasonal components."""
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad_num = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_num, 1)
        end = x[:, -1:, :].repeat(1, pad_num, 1)
        x_padded = torch.cat([front, x, end], dim=1).permute(0, 2, 1)

        trend = self.avg(x_padded).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend

class FourierBlock(nn.Module):
    """Frequency Enhanced Block (FEB) using Fast Fourier Transform (rFFT)."""
    def __init__(self, in_channels, out_channels, seq_len, modes=32):
        super(FourierBlock, self).__init__()
        self.modes = modes
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x):
        B, L, C = x.shape
        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = torch.zeros(B, x_ft.size(1), C, dtype=torch.cfloat, device=x.device)
        modes = min(self.modes, x_ft.size(1))
        out_ft[:, :modes, :] = torch.einsum("bmi,iom->bmo", x_ft[:, :modes, :], self.weights[:, :, :modes])
        x_out = torch.fft.irfft(out_ft, n=L, dim=1)
        return x_out

class FEDformer(nn.Module):
    """FEDformer Long-term Time Series Forecasting Model."""
    def __init__(self, seq_len, label_len, pred_len, enc_in, dec_in, c_out, d_model=64, modes=32):
        super(FEDformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.decomp = MovingAvg(kernel_size=25)
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)

        self.fourier_enc = FourierBlock(d_model, d_model, seq_len, modes=modes)
        self.fourier_dec = FourierBlock(d_model, d_model, label_len + pred_len, modes=modes)

        self.projection = nn.Linear(d_model, c_out)
        self.trend_projection = nn.Linear(enc_in, c_out)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_enc, x_dec):
        seasonal_enc, trend_enc = self.decomp(x_enc)
        enc_out = self.enc_embedding(seasonal_enc)
        enc_out = self.fourier_enc(enc_out)

        dec_out = self.dec_embedding(x_dec)
        attended, _ = self.cross_attention(dec_out, enc_out, enc_out)
        dec_out = self.norm(dec_out + attended)
        dec_out = self.fourier_dec(dec_out)
        seasonal_pred = self.projection(dec_out)

        last_trend = trend_enc[:, -1:, :]
        trend_pred = self.trend_projection(last_trend)
        trend_pred = trend_pred.expand(-1, self.label_len + self.pred_len, -1)
        final_out = seasonal_pred + trend_pred
        return final_out[:, -self.pred_len:, :]

class PatchEmbedding(nn.Module):
    """
    Extracts overlapping local patches from sequence and projects to d_model dimension.
    """
    def __init__(self, patch_len, stride, d_model):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x):
        # x shape: [B * C, L]
        # Unfold sequence into patches -> [B * C, num_patches, patch_len]
        x_patched = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        out = self.projection(x_patched)  # [B * C, num_patches, d_model]
        return out, x_patched.size(1)

class PositionalEncoding(nn.Module):
    """Adds learnable positional encodings to tokenized patch representations."""
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        # x shape: [B * C, num_patches, d_model]
        num_patches = x.size(1)
        return x + self.pos_embed[:, :num_patches, :]

class PatchTST(nn.Module):
    """
    PatchTST Backbone featuring Channel Independence and Local Patching.
    """
    def __init__(self, seq_len, pred_len, num_features, patch_len=16, stride=8, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(PatchTST, self).__init__()
        if patch_len > seq_len:
          raise ValueError("patch_len cannot exceed seq_len")

        if d_model % n_heads != 0:
          raise ValueError("d_model must be divisible by n_heads")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features

        self.patch_embed = PatchEmbedding(patch_len, stride, d_model)
        self.pos_embed = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Calculate number of patches
        num_patches = int((seq_len - patch_len) / stride) + 1

        # Flattened Linear Projection Head
        self.head = nn.Linear(num_patches * d_model, pred_len)

    def forward(self, x):
        # Input shape: [Batch, Seq_Len, Channels]
        B, L, C = x.shape

        # 1. Channel Independence: Flatten Batch & Channels into single dimension
        # x shape becomes: [B * C, Seq_Len]
        x = x.permute(0, 2, 1).reshape(B * C, L)

        # 2. Patch Extraction & Linear Projection -> [B * C, Num_Patches, d_model]
        enc_in, num_patches = self.patch_embed(x)

        # 3. Add Positional Encodings
        enc_in = self.pos_embed(enc_in)

        # 4. Transformer Attention Representation
        enc_out = self.transformer(enc_in)  # [B * C, Num_Patches, d_model]

        # 5. Flatten Patches and Project to Future Prediction Horizon
        enc_out = enc_out.reshape(B * C, -1)  # [B * C, Num_Patches * d_model]
        dec_out = self.head(enc_out)          # [B * C, Pred_Len]

        # 6. Reshape back to original multivariate channel shape
        out = dec_out.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, Pred_Len, C]
        return out

class ProbSparseAttention(nn.Module):
    """Approximate ProbSparse Self-Attention used in Informer."""

    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape

        Q = self.q_proj(queries).view(
            B, L_Q, self.n_heads, self.d_k
        ).transpose(1, 2)

        K = self.k_proj(keys).view(
            B, L_K, self.n_heads, self.d_k
        ).transpose(1, 2)

        V = self.v_proj(values).view(
            B, L_K, self.n_heads, self.d_k
        ).transpose(1, 2)

        # Number of sampled keys and selected queries
        sample_k = min(
            self.factor * int(np.ceil(np.log(L_K + 1))), L_K
        )
        n_top = min(
            self.factor * int(np.ceil(np.log(L_Q + 1))), L_Q
        )

        # Sample keys for each query
        sample_indices = torch.randint(
            L_K, (B, self.n_heads, L_Q, sample_k),
            device=queries.device
        )

        K_expanded = K.unsqueeze(2).expand(
            B, self.n_heads, L_Q, L_K, self.d_k
        )

        sampled_K = torch.gather(
            K_expanded,
            3,
            sample_indices.unsqueeze(-1).expand(
                B, self.n_heads, L_Q, sample_k, self.d_k
            )
        )

        # Estimate query sparsity
        sampled_scores = (
            Q.unsqueeze(3) * sampled_K
        ).sum(dim=-1) / math.sqrt(self.d_k)

        sparsity = (
            sampled_scores.max(dim=-1).values
            - sampled_scores.mean(dim=-1)
        )

        # Select the most important queries
        top_indices = sparsity.topk(n_top, dim=-1).indices

        Q_top = torch.gather(
            Q,
            2,
            top_indices.unsqueeze(-1).expand(
                B, self.n_heads, n_top, self.d_k
            )
        )

        # Attention only for selected queries
        scores = torch.matmul(
            Q_top, K.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        attention = torch.softmax(scores, dim=-1)
        context_top = torch.matmul(attention, V)

        # Mean context for queries not selected
        context = V.mean(dim=2, keepdim=True).expand(
            B, self.n_heads, L_Q, self.d_k
        ).clone()

        context.scatter_(
            2,
            top_indices.unsqueeze(-1).expand(
                B, self.n_heads, n_top, self.d_k
            ),
            context_top
        )

        out = context.transpose(1, 2).contiguous().view(
            B, L_Q, -1
        )

        return self.out_proj(out)

class InformerEncoderLayer(nn.Module):
    """Encoder layer with ProbSparse Attention and Distilling Conv Pooling."""
    def __init__(self, d_model, n_heads):
        super(InformerEncoderLayer, self).__init__()
        self.attn = ProbSparseAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        # Distilling layer (Max Pool along temporal dimension)
        self.distill = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        # Apply distilling pooling
        x_distilled = self.distill(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x_distilled

class Informer(nn.Module):
    """Simplified Informer Model for Multi-Step Forecasting."""
    def __init__(self, seq_len, label_len, pred_len, enc_in, dec_in, c_out, d_model=64, n_heads=4):
      super(Informer, self).__init__()

      if n_heads <= 0:
          raise ValueError("n_heads must be positive")

      if d_model % n_heads != 0:
          raise ValueError("d_model must be divisible by n_heads")

      self.seq_len = seq_len
      self.label_len = label_len
      self.pred_len = pred_len

      self.enc_embedding = nn.Linear(enc_in, d_model)
      self.dec_embedding = nn.Linear(dec_in, d_model)

      self.encoder = InformerEncoderLayer(d_model, n_heads)

      self.cross_attention = nn.MultiheadAttention(
          d_model, n_heads, batch_first=True
      )

      self.norm = nn.LayerNorm(d_model)
      self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_dec):
      enc_out = self.enc_embedding(x_enc)
      enc_out = self.encoder(enc_out)

      dec_out = self.dec_embedding(x_dec)

      cross_out, _ = self.cross_attention(
          dec_out, enc_out, enc_out
      )

      dec_out = self.norm(dec_out + cross_out)

      out = self.projection(dec_out)

      return out[:, -self.pred_len:, :]

class TransformLayer(nn.Module):
    """Learns growth and seasonal transforms via Fourier domain mapping."""
    def __init__(self, d_model, K=8):
        super(TransformLayer, self).__init__()
        self.K = K
        self.d_model = d_model

    def forward(self, x):
        B, L, D = x.shape
        x_ft = torch.fft.rfft(x, dim=1)  # Frequency domain representation

        # Select top K frequencies
        amplitude = torch.abs(x_ft)
        _, top_k_idx = torch.topk(amplitude, min(self.K, x_ft.shape[1]), dim=1)

        # Zero out irrelevant frequencies (Smoothing filter)
        mask = torch.zeros_like(x_ft, dtype=torch.bool)
        mask.scatter_(1, top_k_idx, True)
        x_ft_filtered = x_ft * mask

        # Inverse FFT back to time domain
        return torch.fft.irfft(x_ft_filtered, n=L, dim=1)

class ExponentialSmoothingLayer(nn.Module):
    """Applies dampening exponential smoothing for Level and Trend tracking."""
    def __init__(self, d_model):
        super(ExponentialSmoothingLayer, self).__init__()
        self.alpha = nn.Parameter(torch.randn(d_model))
        self.beta = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        B, L, D = x.shape
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        level = x.new_zeros(B, D)
        growth = x.new_zeros(B, D)

        levels, growths = [], []
        for t in range(L):
            x_t = x[:, t, :]
            next_level = alpha * x_t + (1 - alpha) * (level + growth)
            next_growth = beta * (next_level - level) + (1 - beta) * growth
            level, growth = next_level, next_growth
            levels.append(level.unsqueeze(1))
            growths.append(growth.unsqueeze(1))

        return torch.cat(levels, dim=1), torch.cat(growths, dim=1)

class ETSformerEncoderLayer(nn.Module):
    """ETSformer Encoder: Separates components into Level/Trend and Seasonality."""
    def __init__(self, d_model, K=8):
        super(ETSformerEncoderLayer, self).__init__()
        self.transform = TransformLayer(d_model, K=K)
        self.ets = ExponentialSmoothingLayer(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        # 1. Fourier Frequency Decomposition (Seasonality)
        season = self.transform(x)
        res = x - season

        # 2. Exponential Smoothing (Level and Growth)
        level, growth = self.ets(res)

        # 3. Residual FFN Pass
        res = self.norm1(res)
        res = res + self.ffn(res)
        res = self.norm2(res)

        return res, level, growth, season

class ETSformer(nn.Module):
    """ETSformer Architecture for Time Series Forecasting."""
    def __init__(self, seq_len, pred_len, enc_in, c_out, d_model=64, K=8, e_layers=2):
        super(ETSformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.encoder_layers = nn.ModuleList([
            ETSformerEncoderLayer(d_model, K=K) for _ in range(e_layers)
        ])

        # Projections across temporal dimensions
        self.head_level = nn.Linear(1, pred_len)
        self.head_growth = nn.Linear(1, pred_len)
        self.head_season = nn.Linear(seq_len, pred_len)
        self.head_residual = nn.Linear(seq_len, pred_len)

        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_dec=None):
        res = self.enc_embedding(x_enc)

        total_level = 0
        total_growth = 0
        total_season = 0

        # Stacked ETS Encoder layers
        for layer in self.encoder_layers:
            res, level, growth, season = layer(res)
            total_level = total_level + level[:, -1:, :]   # [B, 1, d_model]
            total_growth = total_growth + growth[:, -1:, :] # [B, 1, d_model]
            total_season = total_season + season            # [B, seq_len, d_model]


        # Project level and growth
        out_level = self.head_level(
            total_level.permute(0, 2, 1)
        ).permute(0, 2, 1)

        out_growth = self.head_growth(
            total_growth.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # Project seasonality
        out_season = self.head_season(
            total_season.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # Use the final residual representation
        out_residual = self.head_residual(
            res.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # Combine all components
        forecast = (
            out_level
            + out_growth
            + out_season
            + out_residual
        )

        output = self.projection(forecast)

        return output

class DSWEmbedding(nn.Module):
    """Dimension-Segment-Wise (DSW) Embedding."""
    def __init__(self, seg_len, d_model):
        super(DSWEmbedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        B, L, D = x.shape
        x = x.permute(0, 2, 1).unfold(dimension=-1, size=self.seg_len, step=self.seg_len)
        out = self.linear(x)
        return out

class TwoStageAttention(nn.Module):
    """Two-Stage Attention (TSA) Layer: Cross-Time & Cross-Dimension."""
    def __init__(self, n_heads, d_model, d_ff=128):
        super(TwoStageAttention, self).__init__()

        if n_heads <= 0:
          raise ValueError("n_heads must be positive")

        if d_model % n_heads != 0:
          raise ValueError("d_model must be divisible by n_heads")

        self.time_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.dim_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        B, D, N, d_model = x.shape

        # Stage 1: Cross-Time Attention
        x_time = x.view(B * D, N, d_model)
        attn_time, _ = self.time_attn(x_time, x_time, x_time)
        x_time = self.norm1(x_time + attn_time).view(B, D, N, d_model)

        # Stage 2: Cross-Dimension Attention
        x_dim = x_time.permute(0, 2, 1, 3).contiguous().view(B * N, D, d_model)
        attn_dim, _ = self.dim_attn(x_dim, x_dim, x_dim)
        x_dim = self.norm2(x_dim + attn_dim).view(B, N, D, d_model).permute(0, 2, 1, 3)

        # Feed-Forward Stage
        out = self.norm3(x_dim + self.ffn(x_dim))
        return out

class Crossformer(nn.Module):
    """Crossformer Architecture for Multivariate Time Series Forecasting."""
    def __init__(self, seq_len, pred_len, in_dim, seg_len=16, d_model=64, n_heads=4, depth=2):
        super(Crossformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.pad_len = (seg_len - (seq_len % seg_len)) % seg_len

        effective_seq_len = seq_len + self.pad_len
        self.num_seg = effective_seq_len // seg_len

        self.embedding = DSWEmbedding(seg_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, in_dim, self.num_seg, d_model))

        self.layers = nn.ModuleList([
            TwoStageAttention(n_heads=n_heads, d_model=d_model) for _ in range(depth)
        ])

        self.head = nn.Linear(self.num_seg * d_model, pred_len)

    def forward(self, x_enc, x_dec=None):
        B, L, D = x_enc.shape

        if self.pad_len > 0:
            pad = x_enc.new_zeros(B, self.pad_len, D)
            x_enc = torch.cat([x_enc, pad], dim=1)

        x_emb = self.embedding(x_enc) + self.pos_emb

        for layer in self.layers:
            x_emb = layer(x_emb)

        x_flat = x_emb.view(B, D, -1)
        out = self.head(x_flat).permute(0, 2, 1)
        return out
