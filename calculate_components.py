import argparse
from datetime import datetime
from pathlib import Path
from pprint import pprint
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA, SVHN, MNIST
from tqdm import tqdm

from models import vae as vanilla_vae
from utils import boundary_loglikelihood


def main():
    parser = argparse.ArgumentParser(description="Runner for VAE model")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="configs/vae_celeba.yaml",
    )
    parser.add_argument(
        "--sigma2",
        "-s",
        type=float,
        dest="sigma2",
        help="sigma^2 for MSE loss to calculate p(x|z)",
        default=None,
    )
    parser.add_argument("--num_repeat", type=int, default=10)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--rate_extended", type=int, default=2)
    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)

    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if args.sigma2 is None:
        try:
            args.sigma2 = float(config["exp_params"]["sigma2"])
            assert args.sigma2 > 0
        except KeyError:
            pass

    pprint(config)

    if not Path(config["results"]["resume_path"]).is_file():
        config["results"]["resume_path"] = "/results" / Path(
            config["results"]["resume_path"]
        )
        config["results"]["root"] = "/results" / Path(config["results"]["root"])

    model = getattr(vanilla_vae, config["name"])(**config["model_params"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[0;1;31m{device=}\033[0m")

    if Path(config["results"]["resume_path"]).is_file():
        print("==> Load checkpoint..")
        model, _, checkpoint = vanilla_vae.load_model(
            model, None, config["results"]["resume_path"], device
        )
    else:
        raise SystemError(
            f"Resuming model does not exist! -> '{config['results']['resume_path']}'"
        )
    model = model.to(device)
    model.eval()

    kwargs = {}
    if device.type == "cuda":
        torch.cuda.manual_seed(config["exp_params"]["manual_seed"])
        kwargs.update(
            {
                "num_workers": config["data_params"]["num_workers"],
                "pin_memory": config["data_params"]["pin_memory"],
            }
        )

    if config["data_params"]["name"] == "celeba":
        trans_data = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(config["data_params"]["patch_size"]),
                transforms.ToTensor(),
            ]
        )
        data_loader = DataLoader(
            CelebA(
                config["data_params"]["data_path"],
                split="test",
                transform=trans_data,
                download=config["data_params"]["download"],
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs,
        )

    elif config["data_params"]["name"] == "svhn":
        trans_data = transforms.Compose(
            [
                transforms.Resize(config["data_params"]["patch_size"]),
                transforms.ToTensor(),
            ]
        )
        data_loader = DataLoader(
            SVHN(
                config["data_params"]["data_path"],
                split="test",
                transform=trans_data,
                download=config["data_params"]["download"],
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs,
        )

    elif config["data_params"]["name"] == "mnist":
        data_loader = DataLoader(
            MNIST(
                config["data_params"]["data_path"],
                train=False,
                transform=transforms.ToTensor(),
                download=config["data_params"]["download"],
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs,
        )

    save_values = f"{config['results']['root']}/{config['name']}_{config['results']['datetime']}/values/{datetime.now().strftime('%Y-%m-%d')}"
    Path(save_values).mkdir(exist_ok=True, parents=True)

    if config["exp_params"]["loss"] == "bce":
        loss_bce = True
        sigma2 = None
    else:
        loss_bce = False
        sigma2 = args.sigma2
    print(f"\033[0;1;33m{loss_bce=}, {sigma2=}\033[0m")

    for nf in tqdm(range(args.num_repeat), desc="Repeat", leave=True):

        np_values2bound_I = np_values2bound_II = None

        with torch.no_grad():
            for ii, (data, _) in tqdm(
                enumerate(data_loader, 1),
                total=len(data_loader),
                desc="Data loop",
                leave=False,
            ):
                data = data.to(device)
                mu, log_var = model.encode(data)

                mu_extended = torch.tile(
                    mu, (args.rate_extended, *(1 for _ in range(1, mu.dim())))
                )
                log_var_extended = torch.tile(
                    log_var, (args.rate_extended, *(1 for _ in range(1, log_var.dim())))
                )
                data_extended = torch.tile(
                    data, (args.rate_extended, *(1 for _ in range(1, data.dim())))
                )

                values2bound_I = values2bound_II = None

                num_iter, rest = divmod(args.n, args.rate_extended)
                num_iter = num_iter + 1 if rest else num_iter
                for _ in tqdm(range(num_iter), desc="N", leave=False):
                    # I time
                    z, s = vanilla_vae.reparameterization(mu_extended, log_var_extended)
                    recons = model.decode(z)
                    output, log_p_xz, log_p_z, log_q_zx = boundary_loglikelihood(
                        data_extended,
                        recons,
                        z,
                        mu_extended,
                        s,
                        loss_bce=loss_bce,
                        sigma2=sigma2,
                    )
                    if values2bound_I is None:
                        values2bound_I = output.view(args.rate_extended, -1)
                    else:
                        values2bound_I = torch.vstack(
                            (values2bound_I, output.view(args.rate_extended, -1))
                        )

                    # II time
                    z, s = vanilla_vae.reparameterization(mu_extended, log_var_extended)
                    recons = model.decode(z)
                    output, log_p_xz, log_p_z, log_q_zx = boundary_loglikelihood(
                        data_extended,
                        recons,
                        z,
                        mu_extended,
                        s,
                        loss_bce=loss_bce,
                        sigma2=sigma2,
                    )
                    if values2bound_II is None:
                        values2bound_II = output.view(args.rate_extended, -1)
                    else:
                        values2bound_II = torch.vstack(
                            (values2bound_II, output.view(args.rate_extended, -1))
                        )

                if np_values2bound_I is None:
                    np_values2bound_I = (
                        values2bound_I.t().detach().cpu().numpy()
                    )  # [bs, n]
                    np_values2bound_II = values2bound_II.t().detach().cpu().numpy()
                else:
                    np_values2bound_I = np.vstack(
                        (np_values2bound_I, values2bound_I.t().detach().cpu().numpy())
                    )
                    np_values2bound_II = np.vstack(
                        (np_values2bound_II, values2bound_II.t().detach().cpu().numpy())
                    )

        filename = f"{save_values}/{str(uuid4())}.npz"
        while Path(filename).is_file():
            filename = f"{save_values}/{str(uuid4())}.npz"
        print(f'\rSave data to file: "{filename}" .....', end="")
        np.savez(
            filename,
            values2bound_I=np_values2bound_I,
            values2bound_II=np_values2bound_II,
        )
    print("DONE")


if __name__ == "__main__":
    main()
