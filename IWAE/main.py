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
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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

    if args.num_epochs is not None:
        config["exp_params"]["epochs"] = args.num_epochs
    if args.lr is not None:
        config["exp_params"]["LR"] = args.lr
    config["results"]["root"] = config["results"]["root"] + "_IWAE"
    config["exp_params"]["num_samples"] = args.num_samples

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
    criterion = vanilla_vae.calc_loss

    scores_dict = {
        "test_loss": [],
        "test_epoch": [],
        "train_loss": [],
        "train_epoch": [],
    }
    saved_epochs = []
    num_stop = 5
    early_stopping = 0

    beta = 0

    writer = SummaryWriter(results_dirs["logs"])
    epoch_tqdm = tqdm(range(config["exp_params"]["epochs"]), desc="Training")
    for epoch in epoch_tqdm:
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
            # ===================forward=====================
            data = data.to(device)
            train_loss = criterion(model, data, beta, **config["exp_params"])

            # ===================backward====================
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()

            # ===================logger========================
            writer.add_scalar("train/loss", train_loss, epoch * len(train_loader) + i)
            trn_tqdm.set_description(f"Train loss: {train_loss:.4f}")
            try:
                scores_dict["train_loss"][epoch] += train_loss.item()
            except IndexError:
                scores_dict["train_loss"].append(train_loss.item())

            if beta < 2:
                beta += 0.001  # Warm-up

        scores_dict[f"train_loss"][epoch] /= len(train_loader)

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
                test_loss = criterion(model, data, beta, **config["exp_params"])

                # ===================logger========================
                writer.add_scalar("test/loss", test_loss, epoch * len(test_loader) + i)
                tst_tqdm.set_description(f"Test loss: {test_loss:.4f}")
                if i == len(test_loader) - 1:
                    data = data[:8]
                    recons, _, _, _ = model(data)
                    vutils.save_image(
                        torch.cat((recons, data), dim=0),
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
                    scores_dict["test_loss"][epoch] += test_loss.item()
                except IndexError:
                    scores_dict["test_loss"].append(test_loss.item())
        scores_dict["test_loss"][epoch] /= len(test_loader)
        ####################################
        #       logger, save model         #
        ####################################
        epoch_tqdm.set_description(
            f"Loss: [{scores_dict[f'train_loss'][epoch]:.4f}|"
            f"{scores_dict['test_loss'][epoch]:.4f}] ({early_stopping:d})"
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
