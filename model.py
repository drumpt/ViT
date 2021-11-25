import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.linear_projection = nn.Linear((patch_size ** 2) * in_channels, emb_size)
        self.class_embedding = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.permute(x, (0, 2, 3, 1)) # b, h, w, c
        x = x.reshape(b, (h // self.patch_size * w // self.patch_size), (self.patch_size ** 2) * c) # b, n, p^2 * c where n = h // p * w // p
        out = self.linear_projection(x) # b, n, 1
        out = torch.cat((self.class_embedding.expand(b, -1, -1), out), 1) # b, (n + 1), 1
        out += self.positional_embedding # b, (n + 1), 1
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 768, num_heads = 12):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = int(self.emb_size / self.num_heads)
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = self.head_size ** -0.5

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, _ = x.shape
        queries = self.queries(x).view(b, n, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        keys = self.keys(x).view(b, n, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        values  = self.values(x).view(b, n, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))  # bhqd x bhdk -> bhqk
        attention_scores = attention_scores * self.scaling
        attention_probs = self.softmax(attention_scores)
        
        out = torch.matmul(attention_probs, values).permute(0, 2, 1, 3).contiguous().view(b, n, self.emb_size)
        out = self.projection(out)
        return out



class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion = 4, dropout = 0.1):
        super(FeedForwardBlock, self).__init__()
        mlp_size = expansion * emb_size
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, emb_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        out = self.mlp(x)

        return x



class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size = 768, num_heads = 12, dropout = 0.1, expansion = 4):
        super(TransformerEncoderBlock, self).__init__()
        
        self.attention_norm = nn.LayerNorm(emb_size)
        self.mlp_norm = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(emb_size, num_heads)
        self.mlp = FeedForwardBlock(emb_size, expansion, dropout)

        self.attn_dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attention(x)
        x = self.attn_dropout(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, layers = 12, emb_size = 768, num_heads = 12, dropout = 0.1, expansion = 4):
        super(TransformerEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.Encoders = nn.Sequential(*[TransformerEncoderBlock(emb_size, num_heads, dropout, expansion) for _ in range(layers)])

    def forward(self, x):
        out = self.Encoders(x)

        return out



class ClassificationHead(nn.Module):
    def __init__(self, emb_size = 768, n_classes = 1000):
        super(ClassificationHead, self).__init__()
        
        self.layernorm = nn.LayerNorm(emb_size)
        self.mlp_head = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = x[:, 0]      # b x n x e -> b x e
        out = self.mlp_head(self.layernorm(x))

        return out



class ViT(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 224, layers = 12, n_classes = 1000, num_heads = 12, dropout = 0.1, expansion = 4):
        super(ViT, self).__init__()
        self.n_classes = n_classes
        self.patchembedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformerencoder = TransformerEncoder(layers, emb_size, num_heads, dropout, expansion)
        self.classificationhead = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        out = self.patchembedding(x)
        out = self.transformerencoder(out)
        out = self.classificationhead(out).view(-1, self.n_classes)
        
        return out