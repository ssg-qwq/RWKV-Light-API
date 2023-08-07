import torch
import torch.nn.functional as F
from torch import nn
from attend import Attend
from einops import rearrange, repeat
from typing import Callable, Optional, List
from rwkv.utils import PIPELINE
from rwkv.model import RWKV


def exists(val):
    return val is not None


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


def eval_decorator(fn):
    """
    训练中推理用的函数
    """

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def l2norm(t):
    return F.normalize(t, dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False)
    )


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            cross_attend=False,
            scale=8,
            flash=True,
            dropout=0.
    ):
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.attend = Attend(
            flash=flash,
            dropout=dropout,
            scale=scale
        )

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
            self,
            x,
            context=None,
            context_mask=None
    ):
        assert not (exists(context) ^ self.cross_attend)

        n = x.shape[-2]
        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b=x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        if exists(context_mask):
            context_mask = repeat(context_mask, 'b j -> b h i j', h=h, i=n)
            context_mask = F.pad(context_mask, (1, 0), value=True)

        out = self.attend(q, k, v, mask=context_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlocks(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            flash=True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash),
                Attention(dim=dim, dim_head=dim_head, heads=heads, cross_attend=True, flash=flash),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x, context=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context=context) + x

            x = ff(x) + x

        return self.norm(x)




class Denseformer(nn.Module):
    def __init__(
            self,
            *,
            dim,  # dim=trunk_len*query_tokens即分片*查询序列长度
            self_cond=False,
            n_embd=768,
            **kwargs
    ):
        super(Denseformer, self).__init__()
        self.dim = dim  # 维度
        self.pos_emb = nn.Embedding(dim, dim)
        self.transformer_blocks = TransformerBlocks(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, dim, bias=False)

        self.text_embed_proj = nn.Linear(n_embd, dim, bias=False) if n_embd != dim else nn.Identity()

        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    def forward(
            self,
            x: Optional[torch.Tensor],
            return_embed=False,
            return_logits=False,
            labels=None,
            ignore_index=0,
            self_cond_embed=None,
            text_embeds: Optional[torch.Tensor] = None
    ):
        device, b, n, t = x.device, *x.shape
        assert exists(x)
        assert exists(text_embeds)

        context = self.text_embed_proj(text_embeds)

        print('emb_shape', x.shape)
        x = x + self.pos_emb(torch.arange(n, device=device))

        # 条件嵌入，疑似多尺度嵌入
        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        embed = self.transformer_blocks(x, context=context)
        print(f"embed_shape:{embed.shape}")

        logits = self.to_logits(embed)
        print(f"logits_shape:{logits.shape}")

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index=ignore_index)

        if not return_logits:
            return loss

        return loss, logits


def extract_rwkv_emb(path: str, layer_name='emb.weight'):
    model = torch.load(path)
    emb = model[layer_name].clone()
    torch.save({'weight': emb}, path.replace('.pth', '.emb'))
    print('save')
    return model


if __name__ == '__main__':
    ###################
    # # 提取embedding
    # # model = extract_rwkv_emb('RWKV-4-World-CHNtuned-0.1B-v1-20230617-ctx4096.pth')
    # model = torch.load('RWKV-4-World-CHNtuned-0.1B-v1-20230617-ctx4096.emb')
    #
    # emb = nn.Embedding(65536, 768)
    # emb.load_state_dict(model)
    #
    # print(emb)
    # a = torch.randint(0, 65536, (1, 256))
    # b = emb(a)
    # print(b, b.shape)
    ###################
    # # 模型测试
    # 一个0.1B的Denseformer模型搭建，这是一个尺度上的Denseformer
    encoder_embd_n = 768
    query_len = 16
    trunk_num = 64
    dim = query_len * trunk_num
    T = Denseformer(dim=dim, depth=6, heads=12, dim_head=64, n_embd=encoder_embd_n,
                    flash=False).bfloat16().cuda()


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(T)
    model_parameter_count = count_parameters(T)
    print(model_parameter_count)
    T.eval()
    a = torch.randn((1, encoder_embd_n, dim)).bfloat16().cuda()
    embd = torch.randn((1, 2, encoder_embd_n)).bfloat16().cuda()  # 这个对应
    b = T(a, text_embeds=embd)
    print(b, b.shape)
