import os
import sys
import argparse
import json
import pickle5 as pickle
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append("/home/atan14/projects/evidential/evi/train")
import hooks
from trainer import Trainer
from metrics import MeanAbsoluteError
from loss import MaeLoss, MseLoss, NllLoss, EvidentialLoss, CombinedLoss
from evi.utils.output import get_outdir, make_dir
from evi.models import SchNet, Ensemble, PaiNN
from evi.data import Dataset, concatenate_dict, split_train_test, split_train_validation_test, collate_dicts


MODEL_DICT = {
    "schnet": SchNet,
    "painn": PaiNN,
}

LOSS_DICT = {
    "mae": MaeLoss,
    "mse": MseLoss,
    "nll": NllLoss,
    "evidential": EvidentialLoss,
}


def get_loss_fn(params):
    energy_loss = LOSS_DICT[params["energy_loss"]](output="energy", **params)
    forces_loss = LOSS_DICT[params["forces_loss"]](output="energy_grad", **params)
    combined_loss = CombinedLoss(
        energy_loss=energy_loss,
        energy_coef=params["energy_coef"],
        forces_loss=forces_loss,
        forces_coef=params["forces_coef"],
    )
    return combined_loss


class TrainPipeline(Trainer):
    def __init__(self, params, restore=False):
        self.params = params
        self.train_params = params.get("train")
        self.model_params = params.get("model")
        self.loss_params = params.get("loss")
        self.dset_params = params.get("dset")
        self.model_dir = params.get("model")["outdir"]
        self._model = self.build_model()

        if not restore:
            self.get_outdir()

        optimizer = self.get_optim(self._model)
        train_metrics = self.get_metrics()
        hooks = self.get_hooks(train_metrics, optimizer)
        self.loss_fn = get_loss_fn(self.loss_params)

        dset = self.load_dset()
        train_loader, val_loader, split_inds = self.get_loaders(dset)
        self.split_inds = split_inds

        super().__init__(
            model_path=self.model_dir,
            model=self._model,
            loss_fn=self.loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=val_loader,
            checkpoint_interval=self.train_params.get("checkpoint_interval", 10),
            hooks=hooks,
            restore=restore,
        )

    def get_outdir(self):
        self.model_dir = get_outdir(
            model_dir=self.model_dir,
            params=self.params,
        )

    def build_model(self):
        num_networks = self.model_params.get("num_networks", 1)
        model_list = []
        for _ in range(num_networks):
            m = MODEL_DICT[self.model_params["model_type"]](self.model_params)
            model_list.append(m)
        model = Ensemble(model_list)
        return model

    def get_optim(self, model):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(trainable_params, lr=self.train_params.get("lr"))
        return optimizer

    def get_metrics(self):
        train_metrics = [
            MeanAbsoluteError("energy"),
            MeanAbsoluteError("energy_grad"),
        ]
        return train_metrics

    def get_hooks(self, train_metrics, optimizer):
        train_hooks = [
            hooks.CSVHook(
                self.model_dir,
                metrics=train_metrics,
            ),
            hooks.PrintingHook(
                self.model_dir,
                metrics=train_metrics,
                separator=" | ",
                time_strf="%M:%S",
                every_n_epochs=self.train_params.get("every_n_epochs", 10),
            ),
            hooks.ReduceLROnPlateauHook(
                optimizer=optimizer,
                patience=30,
                factor=self.train_params.get("lr_factor", 0.5),
                min_lr=self.train_params.get("min_lr", 1e-7),
                window_length=1,
                stop_after_min=True,
            ),
        ]
        return train_hooks

    def load_dset(self):
        if "train_path" not in self.dset_params:
            dset = Dataset.from_file(self.dset_params['path'])
        else:
            dset = Dataset.from_file(self.dset_params['train_path'])
        return dset

    def get_loaders(self, dset):
        if "train_path" not in self.dset_params:
            train, val, test, (train_idx, val_idx, test_idx) = split_train_validation_test(
                dset,
                return_indices=True,
                **self.dset_params,
            )
            split_inds = {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }
        else:
            train, val, (train_idx, val_idx) = split_train_test(
                dset,
                test_size=self.dset_params['val_size'],
                random_state=self.dset_params['random_state'],
            )
            split_inds = {
                "train": train_idx,
                "val": val_idx,
            }
        batch_size = self.dset_params['batch_size']
        train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_dicts)
        val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_dicts)

        return (train_loader, val_loader, split_inds)

    def run(self):
        self.train(
            device=self.train_params.get("device", "cuda:2"),
            n_epochs=self.train_params.get("n_epochs"),
        )

        with open(f"{self.model_dir}/split_inds.pkl", "wb") as f:
            pickle.dump(self.split_inds, f)

        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path", type=str, help='Path to parameter file to run')
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    params = json.load(open(args.params_path, "r"))
    trainer = TrainPipeline(params=params, restore=args.restore)

    if "inbox" in args.params_path:
        new_params_path = args.params_path.replace("inbox", "running")
    else:
        idx = args.params_path.rfind("/")+1
        new_params_path = f"{args.params_path[:idx]}/running/{args.params_path[idx:]}"
    os.rename(args.params_path, make_dir(new_params_path))

    T = trainer.run()

    print(f"Done training on params {args.params_path}.")
