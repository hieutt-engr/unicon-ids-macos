import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchvision.transforms import Compose, Resize, ToTensor
from torch import Tensor
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128, img_size: int = 32):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 128, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 128,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 128, n_classes: int = 5):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
        
class ProjectionHead(nn.Module):
    def __init__(self, dim_in=128, hidden_dim=256, feat_dim=128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )

    def forward(self, x):
        # Xác định đầu vào là từ mô hình ViT
        # Nếu input là [batch_size, sequence_length, embedding_dim]
        if x.dim() == 3:
            bsz, seq_len, emb_dim = x.shape
            # Reshape thành [batch_size * sequence_length, embedding_dim] để đưa qua MLP
            x = x.reshape(bsz * seq_len, emb_dim)
        
        # Áp dụng các tầng MLP trong ProjectionHead
        feat = self.mlp(x)  # Đầu ra sẽ có kích thước [batch_size * sequence_length, feat_dim]
        feat = F.normalize(feat, dim=1)  # Chuẩn hóa các đặc trưng
        
        # Reshape lại để phù hợp với yêu cầu của UniConLoss hoặc phần tính toán tiếp theo
        return feat.view(bsz, seq_len, -1)  # Đưa về lại [batch_size, sequence_length, feat_dim]

        
class ConViT(nn.Module):
    """ViT encoder + projection head + classifier head"""

    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 4,
                emb_size: int = 128,
                img_size: int = 32,
                depth: int = 12,
                head: str = 'mlp',
                feat_dim: int = 128,
                n_classes: int = 5,
                **kwargs):
        super(ConViT, self).__init__()
        
        # Sử dụng Patch Embedding và Transformer Encoder để làm ViT encoder
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer_encoder = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        self.classifier_head = ClassificationHead(emb_size, n_classes)
        
        # Sử dụng projection head tương tự như ConResNet
        if head == 'linear':
            self.projection_head = nn.Linear(emb_size, feat_dim)
        elif head == 'mlp':
            self.projection_head = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.ReLU(inplace=True),
                nn.Linear(emb_size, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
        # Sử dụng classifier head cho CE Loss

    def forward(self, x, use_projection=False):
        # Bước qua Patch Embedding và Transformer Encoder
        x = self.patch_embedding(x)  # Đầu vào: [batch_size, 3, 32, 32]
        x = self.transformer_encoder(x)  # Đầu ra: [batch_size, seq_len, emb_size]
        
        # Lấy trung bình các đặc trưng để thu gọn thành vector
        cls_token = x[:, 0]  # Đưa về dạng [batch_size, emb_size]

        # Áp dụng projection head nếu được yêu cầu
        if use_projection:
            return F.normalize(self.projection_head(cls_token), dim=1)

        # Ngược lại, sử dụng classifier head cho CE Loss
        return self.classifier_head(x)


        
