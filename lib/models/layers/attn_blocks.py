import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """  # 这段代码实现了候选消除（candidate elimination）的功能，用于在注意力机制中减少计算量和噪声，通过保留注意力权重较高的特征并移除注意力权重较低的特征。
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t  # 提取搜索区域的块数
    bs, hn, _, _ = attn.shape  # 提取bs和注意力头数

    lens_keep = math.ceil(keep_ratio * lens_s)  # 计算要保留的搜索区域的块数
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]  # 提取模板対搜索区域的注意力权重 因为attn是64个模板块+256个搜索块 所以:lens_t是提取了64个模板块， lens_t:提取了256个搜索块 (bs,hn,64,256) 得到的就是模板块対搜索块的注意力权重

    if box_mask_z is not None:  # box_mask_z模板掩码存在
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s 首先沿着第二维求均值得到(B,H,L_s)表示搜索区域的块対所有模板块的注意力权重 然后沿着第一维求均值得到(B,L_s)表示所有头中，搜索区域的块対所有模板块的注意力权重。也就是说最后整合了每个注意力头中搜索区域的块対模板区域的注意力权重。

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)  # 得到搜索区域中要保留的索引和要丢掉的索引(注意力权重大的保留)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]  # 得到模板区域和搜索区域的注意力输出
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C) 这里attentive_tokens(B,K,C)K是要保留的搜索区域的个数，得到搜索区域保留的特征
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0) 这里将模板区域的特征和搜索区域保留的特征堆叠在一起形成新的特征(B,K+64,768)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # nn.Identity()空网络层，占位作用
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)  # 这里的mask是None，x是模板和搜索的所有块的特征 返回通过transformer的每一个块的特征x_attn和关系矩阵(Q*K^-1/C^-2) attn
        x = x + self.drop_path(x_attn)  # 注意力的输出和x相加
        lens_t = global_index_template.shape[1]  # 模板的块数

        removed_index_search = None  # self.keep_ratio_search = 1 keep_ratio_search = 1
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
            # 上面的x是模板区域的特征和搜索区域通过候选消除保留的特征的堆叠(B,K+64,768) 然后将搜索区域中保留的index赋值给global_index_search，搜索区域中删除的index赋值给removed_index_search，这里有个问题，你每次删除的index直接赋值给removed_index_search，那每一次都会覆盖，那么如果有多个transfomer块的话，在最后恢复保留块的位置的时候好像不需要删除的块的index，只需要保留块的index，那这样的话可以正确恢复，那这个删除块的index有啥用呢？记录用，删除掉没关系，外面有一个列表来append这个值，不会覆盖而是添加
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 候选消融和结果x通过前馈神经网络再和x进行残差连接 这样得到的结果x就是一个transformer块的输出了
        return x, global_index_template, global_index_search, removed_index_search, attn  # 这里的attn是原始的搜索区域和模板区域的注意力权重


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
