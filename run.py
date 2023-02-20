import argparse
from datetime import datetime
from pathlib import Path
from pprint import pprint
from random import randint

import numpy as np
import torch
import torchvision.utils as vutils
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CelebA, SVHN, MNIST
from tqdm import tqdm

from models import vae as vanilla_vae


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
    args = parser.parse_args()
    print(args)

    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    fit_model(config)


def fit_model(config: dict) -> None:
    config["results"]["datetime"] = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dirs = {
        "root": f"{config['results']['root']}/{config['name']}_{config['results']['datetime']}"
    }
    for name in ["logs", "sample", "reconstruction", "checkpoint"]:
        results_dirs[name] = f"{results_dirs['root']}/{name}"
        Path(results_dirs[name]).mkdir(exist_ok=True, parents=True)

    if config["exp_params"]["manual_seed"] is None:
        config["exp_params"]["manual_seed"] = randint(10, 10000)
    torch.manual_seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])
    pprint(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[0;1;31m{device=}\033[0m")

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
        transforms_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(config["data_params"]["patch_size"]),
                transforms.ToTensor(),
            ]
        )
        transforms_test = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(config["data_params"]["patch_size"]),
                transforms.ToTensor(),
            ]
        )
        train_loader = DataLoader(
            CelebA(
                config["data_params"]["data_path"],
                split="train",
                transform=transforms_train,
                download=config["data_params"]["download"],
            ),
            batch_size=config["data_params"]["train_batch_size"],
            shuffle=True,
            **kwargs,
        )
        test_loader = DataLoader(
            CelebA(
                config["data_params"]["data_path"],
                split="test",
                transform=transforms_test,
                download=config["data_params"]["download"],
            ),
            batch_size=config["data_params"]["val_batch_size"],
            shuffle=False,
            **kwargs,
        )
    elif config["data_params"]["name"] == "svhn":
        transforms_data = transforms.Compose(
            [
                transforms.Resize(config["data_params"]["patch_size"]),
                transforms.ToTensor(),
            ]
        )
        train_loader = DataLoader(
            SVHN(
                config["data_params"]["data_path"],
                split="train",
                transform=transforms_data,
                download=config["data_params"]["download"],
            ),
            batch_size=config["data_params"]["train_batch_size"],
            shuffle=True,
            **kwargs,
        )
        test_loader = DataLoader(
            SVHN(
                config["data_params"]["data_path"],
                split="test",
                transform=transforms_data,
                download=config["data_params"]["download"],
            ),
            batch_size=config["data_params"]["val_batch_size"],
            shuffle=False,
            **kwargs,
        )
    elif config["data_params"]["name"] == "mnist":
        train_loader = DataLoader(
            MNIST(
                config["data_params"]["data_path"],
                train=True,
                transform=transforms.ToTensor(),
                download=config["data_params"]["download"],
            ),
            batch_size=config["data_params"]["train_batch_size"],
            shuffle=True,
            **kwargs,
        )
        test_loader = DataLoader(
            MNIST(
                config["data_params"]["data_path"],
                train=False,
                transform=transforms.ToTensor(),
                download=config["data_params"]["download"],
            ),
            batch_size=config["data_params"]["val_batch_size"],
            shuffle=False,
            **kwargs,
        )

    model = getattr(vanilla_vae, config["name"])(**config["model_params"])
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["exp_params"]["LR"],
        weight_decay=config["exp_params"]["weight_decay"],
    )
    criterion = getattr(vanilla_vae, f"loss_{config['exp_params']['loss']}")
    scheduler = None
    if config["exp_params"]["scheduler_gamma"] is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config["exp_params"]["scheduler_gamma"]
        )

    scores_dict = {
        "test_loss": [],
        "test_reconstruction": [],
        "test_kld": [],
        "test_epoch": [],
        "train_loss": [],
        "train_reconstruction": [],
        "train_kld": [],
        "train_epoch": [],
    }
    saved_epochs = []
    num_stop = 5
    early_stopping = 0

    writer = SummaryWriter(results_dirs["logs"])
    epoch_tqdm = tqdm(range(config["exp_params"]["epochs"]), desc="Training")
    for epoch in epoch_tqdm:
        log_dict = {}
        ####################################
        #            train step            #
        ####################################
        model.train()

        scores_dict["train_epoch"].append(epoch)
        trn_tqdm = tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            desc="Train step",
            leave=False,
        )
        for i, (data, _) in trn_tqdm:
            data = data.to(device)

            # ===================forward=====================
            recons, _, mu, log_var = model(data)
            train_loss = criterion(recons, data, mu, log_var, **config["exp_params"])

            # ===================backward====================
            optimizer.zero_grad(set_to_none=True)
            train_loss["loss"].backward()
            optimizer.step()

            # ===================logger========================
            writer.add_scalar(
                "train/loss", train_loss["loss"], epoch * len(train_loader) + i
            )
            writer.add_scalar(
                "train/reconstruction",
                train_loss["reconstruction"],
                epoch * len(train_loader) + i,
            )
            writer.add_scalar(
                "train/kld", train_loss["kld"], epoch * len(train_loader) + i
            )

            trn_tqdm.set_description(
                f"Train loss: {train_loss['loss']:.4f}, recon: {train_loss['reconstruction']:.4f}, "
                f"kld: {train_loss['kld']:.2f}"
            )
            if i == len(train_loader) - 1:
                log_dict.update(
                    {f"trn_{key}": val.item() for key, val in train_loss.items()}
                )

            try:
                for name in ["loss", "reconstruction", "kld"]:
                    scores_dict[f"train_{name}"][epoch] += train_loss[name].item()
            except IndexError:
                for name in ["loss", "reconstruction", "kld"]:
                    scores_dict[f"train_{name}"].append(train_loss[name].item())
        for name in ["loss", "reconstruction", "kld"]:
            scores_dict[f"train_{name}"][epoch] /= len(train_loader)
        if scheduler is not None:
            writer.add_scalar(
                "train/learning_rate", optimizer.param_groups[0]["lr"], epoch
            )

        ####################################
        #          validation step         #
        ####################################
        model.eval()

        scores_dict["test_epoch"].append(epoch)
        tst_tqdm = tqdm(
            enumerate(test_loader, 0),
            total=len(test_loader),
            desc="Test step",
            leave=False,
        )
        with torch.no_grad():
            for i, (data, _) in tst_tqdm:
                data = data.to(device)

                recons, _, mu, log_var = model(data)
                test_loss = criterion(recons, data, mu, log_var, **config["exp_params"])

                # ===================logger========================
                writer.add_scalar(
                    "test/loss", test_loss["loss"], epoch * len(test_loader) + i
                )
                writer.add_scalar(
                    "test/reconstruction",
                    test_loss["reconstruction"],
                    epoch * len(test_loader) + i,
                )
                writer.add_scalar(
                    "test/kld", test_loss["kld"], epoch * len(test_loader) + i
                )

                tst_tqdm.set_description(
                    f"Test loss: {test_loss['loss']:.4f}, "
                    f"recon: {test_loss['reconstruction']:.4f}, kld: {test_loss['kld']:.2f}"
                )
                if i == len(test_loader) - 1:
                    log_dict.update(
                        {f"tst_{key}": val.item() for key, val in test_loss.items()}
                    )

                    vutils.save_image(
                        torch.cat((recons[:8], data[:8]), dim=0),
                        f"{results_dirs['reconstruction']}/epoch_{epoch}.png",
                        normalize=True,
                        nrow=8,
                    )

                    samples = model.sample(16, device)
                    vutils.save_image(
                        samples,
                        f"{results_dirs['sample']}/epoch_{epoch}.png",
                        normalize=True,
                        nrow=4,
                    )
                try:
                    for name in ["loss", "reconstruction", "kld"]:
                        scores_dict[f"test_{name}"][epoch] += test_loss[name].item()
                except IndexError:
                    for name in ["loss", "reconstruction", "kld"]:
                        scores_dict[f"test_{name}"].append(test_loss[name].item())
        for name in ["loss", "reconstruction", "kld"]:
            scores_dict[f"test_{name}"][epoch] /= len(test_loader)
        ####################################
        #       logger, save model         #
        ####################################
        epoch_tqdm.set_description(
            f"Loss: [{log_dict['trn_loss']:.4f}|{log_dict['tst_loss']:.4f}], "
            f"recon: [{log_dict['trn_reconstruction']:.4f}|{log_dict['tst_reconstruction']:.4f}], "
            f"kld: [{log_dict['trn_kld']:.2f}|{log_dict['tst_kld']:.2f}] ({early_stopping:d})"
        )

        # save the model
        if (
            np.mean(scores_dict["test_loss"][-(num_stop + 1) :])
            >= scores_dict["test_loss"][-1]
        ):
            vanilla_vae.save_model(
                model,
                None,
                f"{results_dirs['checkpoint']}/models_epoch{epoch:03}.pth",
                epoch=epoch,
            )
            saved_epochs.append(epoch)
            tmp = scores_dict["test_epoch"][np.argmin(scores_dict["test_loss"])]
            for id_ep in np.setdiff1d(np.array(saved_epochs), [tmp]):
                Path(f"{results_dirs['checkpoint']}/models_epoch{id_ep:03}.pth").unlink(
                    missing_ok=True
                )
                # saved_epochs.remove(id_ep)
            early_stopping = 0
        else:
            early_stopping += 1

        if early_stopping == num_stop:
            print(f"\033[0;1;31mEarly stopping!\033[0m")
            break

        ####################################
        #           scheduler              #
        ####################################
        if scheduler is not None:
            scheduler.step()

    writer.close()

    config["train_step"] = False
    tmp = scores_dict["test_epoch"][np.argmin(scores_dict["test_loss"])]
    config["results"][
        "resume_path"
    ] = f"{results_dirs['checkpoint']}/models_epoch{tmp:03}.pth"
    with open(f"{results_dirs['root']}/config.yaml", "w") as yaml_file:
        yaml.safe_dump(
            config,
            yaml_file,
            default_style=None,
            default_flow_style=False,
            sort_keys=False,
        )


if __name__ == "__main__":
    main()
