# try:
#     from kat_rational import KAT_Group
# except ImportError as e:
#     pass
# import torch
# from torch import nn
# class KAN(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(
#             self,
#             in_features,
#             hidden_features=None,
#             out_features=None,
#             act_layer=None,
#             norm_layer=None,
#             bias=True,
#             drop=0.,
#             use_conv=False,
#             act_init="gelu",
#     ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         bias = to_2tuple(bias)
#         drop_probs = to_2tuple(drop)
#         linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

#         self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
#         self.act1 = KAT_Group(mode="identity")
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
#         self.act2 = KAT_Group(mode=act_init)
#         self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
#         self.drop2 = nn.Dropout(drop_probs[1])

#     def forward(self, x):
#         x = self.act1(x)
#         x = self.drop1(x)
#         x = self.fc1(x)
#         x = self.act2(x)
#         x = self.drop2(x)
#         x = self.fc2(x)
#         return x

# class KatAttention(nn.Module):
#     fused_attn: Final[bool]

#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             qkv_bias: bool = False,
#             qk_norm: bool = False,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.,
#             norm_layer: nn.Module = nn.LayerNorm,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.fused_attn = use_fused_attn()

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)

#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(
#                 q, k, v,
#                 dropout_p=self.attn_drop.p if self.training else 0.,
#             )
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class LayerScale(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             init_values: float = 1e-5,
#             inplace: bool = False,
#     ) -> None:
#         super().__init__()
#         self.inplace = inplace
#         self.gamma = nn.Parameter(init_values * torch.ones(dim))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x.mul_(self.gamma) if self.inplace else x * self.gamma

# class Kat(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int=8,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = False,
#             qk_norm: bool = False,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             init_values: Optional[float] = None,
#             drop_path: float = 0.,
#             act_layer: nn.Module = nn.GELU,
#             norm_layer: nn.Module = nn.LayerNorm,
#             mlp_layer: nn.Module = KAN,
#             act_init: str = 'gelu',
#     ) -> None:
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = KatAttention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#         )
#         self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         self.mlp = mlp_layer(
#             in_features=dim,
#             hidden_features=int(dim * mlp_ratio),
#             act_layer=act_layer,
#             drop=proj_drop,
#             act_init=act_init,
#         )
#         self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         N, C, H, W = x.size()
#         x = x.flatten(2).permute(0, 2, 1)
#         x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x.permute(0, 2, 1).view([-1, C, H, W]).contiguous()

# class C3k_Kat(C3k):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
#         super().__init__(c1, c2, n, shortcut, g, e, k)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(Kat(c_) for _ in range(n)))

# class C3k2_Kat(C3k2):
#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         super().__init__(c1, c2, n, c3k, e, g, shortcut)
#         self.m = nn.ModuleList(C3k_Kat(self.c, self.c, 2, shortcut, g) if c3k else Kat(self.c) for _ in range(n))

# class Faster_Block_KAN(nn.Module):
#     def __init__(self,
#                  inc,
#                  dim,
#                  n_div=4,
#                  mlp_ratio=2,
#                  drop_path=0.1,
#                  layer_scale_init_value=0.0,
#                  pconv_fw_type='split_cat'
#                  ):
#         super().__init__()
#         self.dim = dim
#         self.mlp_ratio = mlp_ratio
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.n_div = n_div

#         self.mlp = KAN(dim, hidden_features=int(dim * mlp_ratio))

#         self.spatial_mixing = Partial_conv3(
#             dim,
#             n_div,
#             pconv_fw_type
#         )
        
#         self.adjust_channel = None
#         if inc != dim:
#             self.adjust_channel = Conv(inc, dim, 1)

#         if layer_scale_init_value > 0:
#             self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             self.forward = self.forward_layer_scale
#         else:
#             self.forward = self.forward

#     def forward(self, x):
#         N, C, H, W = x.size()
#         if self.adjust_channel is not None:
#             x = self.adjust_channel(x)
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(self.mlp(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous())
#         return x

#     def forward_layer_scale(self, x):
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(
#             self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
#         return x

# class C3k_Faster_KAN(C3k):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
#         super().__init__(c1, c2, n, shortcut, g, e, k)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(Faster_Block_KAN(c_, c_) for _ in range(n)))

# class C3k2_Faster_KAN(C3k2):
#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         super().__init__(c1, c2, n, c3k, e, g, shortcut)
#         self.m = nn.ModuleList(C3k_Faster_KAN(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block_KAN(self.c, self.c) for _ in range(n))
