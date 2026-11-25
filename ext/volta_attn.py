import torch, volta_ext
def attn(q, k, v, P, lens=None, sq=None, sk=None):
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, sq, sk)):
        raise NotImplementedError("kernel 1 backward is step 3")
    return volta_ext.attn_fwd(q, k, v, P, lens, sq, sk)
