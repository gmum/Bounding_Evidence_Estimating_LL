from math import pi, log
import torch


def boundary_loglikelihood(
    img, recon_img, z, mu_z, sigma_z, loss_bce=True, sigma2=None
):
    dim = mu_z.shape[-1]
    tmp = dim * log(2 * pi)

    # q(z|x)
    x_mu = z - mu_z
    log_detsigma = torch.log(sigma_z).mul(2).sum(1)
    log_q_zx = torch.mul(
        tmp
        + log_detsigma
        + torch.einsum(
            "bd,bd->b", torch.square(x_mu), torch.reciprocal(torch.square(sigma_z))
        ),
        -0.5,
    )
    # P(z)
    log_p_z = torch.mul(tmp + torch.einsum("bd,bd->b", z, z), -0.5)
    # p(x|z)
    if loss_bce:
        y = torch.clamp(recon_img, min=1e-8, max=1 - 1e-8)
        log_p_xz = torch.sum(
            img * torch.log(y) + (1 - img) * torch.log(1 - y),
            dim=tuple(range(1, img.dim())),
        )
    else:
        assert sigma2 > 0
        dim = torch.tensor(img.shape[1:], device=img.device).prod()
        tmp = dim.mul(log(2 * pi * sigma2))
        x_mu = img - recon_img
        log_p_xz = torch.mul(
            tmp
            + torch.div(torch.square(x_mu), sigma2).sum(dim=tuple(range(1, img.dim()))),
            -0.5,
        )
    return log_p_xz + log_p_z - log_q_zx, log_p_xz, log_p_z, log_q_zx


def logsubexp(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    test = torch.ge(x, v)
    assert test.all(), f"{x[test]}, {v[test]}, eq: {torch.allclose(x, v)}"
    return torch.where(
        torch.isneginf(v), x, torch.add(x, torch.log1p(torch.exp(v - x).neg()))
    )


def log_variance(
    logsumexp_x: torch.Tensor, logsumexp_2x: torch.Tensor, n: int, unbiased: bool = True
) -> torch.Tensor:
    x = logsumexp_2x  # log(nE(x^2))
    y = logsumexp_x.mul(2) - log(n)  # log(n[E(x)]^2)
    res = logsubexp(x, y)
    if unbiased:
        return res - log(n - 1)
    else:
        return res - log(n)
