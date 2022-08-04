import numpy as np
import torch
from ase import Atoms
from ase.calculators.emt import EMT

from amptorch.trainer import AtomsTrainer

### Construct test data
distances = np.linspace(2, 5, 100)
images = []
for dist in distances:
    image = Atoms(
        "CuCO",
        [
            (-dist * np.sin(0.65), dist * np.cos(0.65), 0),
            (0, 0, 0),
            (dist * np.sin(0.65), dist * np.cos(0.65), 0),
        ],
    )
    image.set_cell([10, 10, 10])
    image.wrap(pbc=True)
    image.set_calculator(EMT())
    images.append(image)

### Construct parameters
# define sigmas
nsigmas = 4
sigmas = np.linspace(0, 2.0, nsigmas + 1, endpoint=True)[1:]

# define MCSH orders
MCSHs_index = 2
MCSHs_dict = {
    0: {
        "orders": [0],
        "sigmas": sigmas,
    },
    1: {
        "orders": [0, 1],
        "sigmas": sigmas,
    },
    2: {
        "orders": [0, 1, 2],
        "sigmas": sigmas,
    },
    # 3: { "orders": [0,1,2,3], "sigmas": sigmas,},
    # 4: { "orders": [0,1,2,3,4], "sigmas": sigmas,},
    # 5: { "orders": [0,1,2,3,4,5], "sigmas": sigmas,},
    # 6: { "orders": [0,1,2,3,4,5,6], "sigmas": sigmas,},
    # 7: { "orders": [0,1,2,3,4,5,6,7], "sigmas": sigmas,},
    # 8: { "orders": [0,1,2,3,4,5,6,7,8], "sigmas": sigmas,},
    # 9: { "orders": [0,1,2,3,4,5,6,7,8,9], "sigmas": sigmas,},
}
MCSHs = MCSHs_dict[MCSHs_index]  # MCSHs is now just the order of MCSHs.

GMP = {
    "MCSHs": MCSHs,
    "atom_gaussians": {
        "C": "amptorch/tests/GMP_params/C_pseudodensity_4.g",
        "O": "amptorch/tests/GMP_params/O_pseudodensity_4.g",
        "Cu": "amptorch/tests/GMP_params/Cu_pseudodensity_4.g",
    },
    "cutoff": 12,
}

elements = ["Cu", "C", "O"]


def get_config():
    config = {
        "model": {
            "name": "singlenn",
            "get_forces": True,
            "num_layers": 3,
            "num_nodes": 20,
            "batchnorm": True,
            "activation": torch.nn.Tanh,
        },
        "optim": {
            "force_coefficient": 0.04,
            "lr": 1e-3,
            "batch_size": 16,
            "epochs": 300,
            "loss": "mse",
            "metric": "mae",
        },
        "dataset": {
            "raw_data": images,
            "fp_scheme": "gmpordernorm",
            "fp_params": GMP,
            "elements": elements,
            "save_fps": True,
            "scaling": {"type": "normalize", "range": (0, 1)},
            "val_split": 0,
        },
        "cmd": {
            "debug": False,
            "run_dir": "./",
            "seed": 1,
            "identifier": "test",
            "verbose": False,
            # Weights and Biases used for logging - an account(free) is required
            "logger": False,
        },
    }

    return config


true_energies = np.array([image.get_potential_energy() for image in images])
true_forces = np.concatenate(np.array([image.get_forces() for image in images]))


def get_energy_metrics(config):
    trainer = AtomsTrainer(config)
    trainer.train()
    predictions = trainer.predict(images)
    pred_energies = np.array(predictions["energy"])
    mae = np.mean(np.abs(true_energies - pred_energies))
    assert mae < 0.03


def get_force_metrics(config):
    trainer = AtomsTrainer(config)
    trainer.train()
    predictions = trainer.predict(images)
    pred_energies = np.array(predictions["energy"])
    pred_forces = np.concatenate(np.array(predictions["forces"]))

    e_mae = np.mean(np.abs(true_energies - pred_energies))
    f_mae = np.mean(np.abs(pred_forces - true_forces))

    assert e_mae < 0.06
    assert f_mae < 0.10


def test_training_gmp():
    torch.set_num_threads(1)

    ### train only
    # energy+forces+mse loss
    config = get_config()
    config["model"]["get_forces"] = True
    config["optim"]["force_coefficient"] = 0.04
    config["optim"]["loss"] = "mse"
    get_force_metrics(config)
    print("Train energy+forces success!")
    # energy+mae loss
    config = get_config()
    config["model"]["get_forces"] = False
    config["optim"]["force_coefficient"] = 0
    config["optim"]["loss"] = "mae"
    get_energy_metrics(config)
    print("Train energy only success!")

    ### train+val
    # energy only
    config = get_config()
    config["model"]["get_forces"] = False
    config["optim"]["force_coefficient"] = 0
    config["optim"]["loss"] = "mae"
    config["dataset"]["val_split"] = 0.1
    get_energy_metrics(config)
    print("Val energy only success!")

    # energy+forces
    config = get_config()
    config["model"]["get_forces"] = True
    config["optim"]["force_coefficient"] = 0.04
    config["optim"]["loss"] = "mse"
    config["dataset"]["val_split"] = 0.1
    get_force_metrics(config)
    print("Val energy+forces success!")


if __name__ == "__main__":
    print("\n\n--------- GMP Training Test ---------\n")
    test_training_gmp()
