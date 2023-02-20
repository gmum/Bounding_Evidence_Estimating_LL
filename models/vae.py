from typing import Tuple, Any
from math import pi, log

import torch
from torch import nn
from torch.nn import functional as F


def reparameterization(mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[Any, Any]:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu), std


def loss_bce(recons, input_x, mu, log_var, **kwargs) -> dict:
    recons_loss = (
        F.binary_cross_entropy(recons, input_x, reduction="none").sum(1).mean()
    )
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kwargs["kld_weight"] * kld_loss
    return {
        "loss": loss,
        "reconstruction": recons_loss.detach(),
        "kld": -kld_loss.detach(),
    }


def loss_mse(recons, input_x, mu, log_var, **kwargs) -> dict:
    recons_loss = F.mse_loss(recons, input_x, reduction="none").sum(1).mean()
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kwargs["kld_weight"] * kld_loss
    return {
        "loss": loss,
        "reconstruction": recons_loss.detach(),
        "kld": -kld_loss.detach(),
    }


def loss_our(recons, input_x, mu, log_var, **kwargs) -> dict:
    dim = torch.tensor(input_x.shape[1:], device=input_x.device).prod()
    tmp = dim.mul(log(2 * pi * kwargs["sigma2"]))
    x_mu = torch.sub(input_x, recons)
    recons_loss = (
        torch.mul(
            tmp
            + torch.div(torch.square(x_mu), kwargs["sigma2"]).sum(
                dim=tuple(range(1, input_x.dim()))
            ),
            -0.5,
        )
        .mean()
        .neg()
    )

    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kwargs["kld_weight"] * kld_loss
    return {
        "loss": loss,
        "reconstruction": recons_loss.detach(),
        "kld": -kld_loss.detach(),
    }


def save_model(model, optimizer, path, **kwargs):
    dict_save = {"model_state_dict": model.state_dict()}
    if optimizer is None:
        dict_save["optimizer_state_dict"] = optimizer
    else:
        dict_save["optimizer_state_dict"] = optimizer.state_dict()
    dict_save.update(kwargs)
    torch.save(dict_save, path)


def load_model(model, optimizer, path, device):
    if device.type == "cuda":
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    del checkpoint["optimizer_state_dict"], checkpoint["model_state_dict"]

    model.eval()
    print(f"\033[0;32mLoad model form: {path}\033[0m")
    return model, optimizer, checkpoint


class ViewLastDim(nn.Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], *self.shape)

    def __repr__(self):
        return f"ViewLastDim{self.shape}"


class VAEceleba(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_dim):
        super(VAEceleba, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_dim = latent_dim

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        self.e3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        self.e4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.e5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf * 8)

        self.fc1 = nn.Linear(ndf * 8 * 4 * 4, latent_dim)
        self.fc2 = nn.Linear(ndf * 8 * 4 * 4, latent_dim)

        # decoder
        self.d1 = nn.Linear(latent_dim, ngf * 8 * 2 * 4 * 4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf * 8 * 2, ngf * 8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf * 8, 1.0e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf * 8, ngf * 4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf * 4, 1.0e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf * 4, ngf * 2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf * 2, 1.0e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf * 2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.0e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> Tuple[Any, Any]:
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf * 8 * 4 * 4)
        return self.fc1(h5), self.fc2(h5)  # mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf * 8 * 2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def forward(self, x: torch.Tensor) -> Tuple[Any, Any, Any, Any]:
        mu, logvar = self.encode(x)
        z, _ = reparameterization(mu, logvar)
        return self.decode(z), z, mu, logvar

    def sample(self, num_samples: int, device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class VAEsvhn(nn.Module):
    def __init__(self, image_channels, hidden_size, latent_dim):
        super(VAEsvhn, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(start_dim=-3, end_dim=-1),
        )

        self.encoder_mean = nn.Linear(hidden_size, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            ViewLastDim(1024, 1, 1),
            nn.ConvTranspose2d(hidden_size, 128, 5, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, image_channels, 6, 2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[Any, Any]:
        x = self.encoder(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        return mean, log_var  # mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[Any, Any, Any, Any]:
        mu, logvar = self.encode(x)
        z, _ = reparameterization(mu, logvar)
        return self.decode(z), z, mu, logvar

    def sample(self, num_samples: int, device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class VAEmnist(nn.Module):
    def __init__(self, latent_dim):
        super(VAEmnist, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
        )

        self.encoder_mean = nn.Linear(256, latent_dim)
        self.encoder_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Sigmoid(),
            ViewLastDim(1, 28, 28),
        )

    def encode(self, x: torch.Tensor) -> Tuple[Any, Any]:
        x = self.encoder(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        return mean, log_var  # mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[Any, Any, Any, Any]:
        mu, logvar = self.encode(x)
        z, _ = reparameterization(mu, logvar)
        return self.decode(z), z, mu, logvar

    def sample(self, num_samples: int, device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


def calc_loss(net, x, beta, num_samples, sigma2, **kwargs):
    x = x.tile(
        num_samples, 1, 1, 1
    )  # make num_samples copies, shape: [n * bs, c, w, h]

    mu, log_var = net.encode(x)

    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    # z = eps.mul(std).add_(mu)
    z_mu = eps.mul(std)
    z = z_mu.add(mu)
    recon_x = net.decode(z)

    std = std.view(num_samples, -1, *std.shape[1:])
    z_mu = z_mu.view(num_samples, -1, *z_mu.shape[1:])
    z = z.view(num_samples, -1, *z.shape[1:])
    x = x.view(num_samples, -1, *x.shape[1:])
    recon_x = recon_x.view(num_samples, -1, *recon_x.shape[1:])

    dim = z.shape[-1]
    tmp = dim * log(2 * pi)

    # q(z|x)
    log_detsigma = torch.log(std).mul(2).sum(-1)
    log_QzGx = torch.mul(
        tmp
        + log_detsigma
        + torch.einsum(
            "ebd,ebd->eb", torch.square(z_mu), torch.reciprocal(torch.square(std))
        ),
        -0.5,
    )
    # p(z)
    log_Pz = torch.mul(tmp + torch.einsum("ebd,ebd->eb", z, z), -0.5)
    # p(x|z)
    assert sigma2 > 0
    dim = torch.tensor(x.shape[-3:], device=x.device).prod()
    tmp = dim.mul(log(2 * pi * sigma2))
    x_mu = x - recon_x
    log_PxGz = torch.mul(
        tmp + torch.div(torch.square(x_mu), sigma2).sum(dim=tuple(range(2, x.dim()))),
        -0.5,
    )

    log_Pxz = log_Pz + log_PxGz  # log(p(x, z))

    # Weighting according to equation 13 from IWAE paper
    log_weight = (log_Pxz - log_QzGx).detach().data
    log_weight = log_weight - torch.max(log_weight, 0)[0]
    weight = torch.exp(log_weight)
    weight = weight / torch.sum(weight, 0)

    # scaling
    return -torch.mean(torch.sum(weight * (log_PxGz + (log_Pz - log_QzGx) * beta), 0))
