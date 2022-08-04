import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from amptorch.trainer import AtomsTrainer

distances = np.linspace(2, 5, 10)
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
        "C": "./valence_gaussians/C_pseudodensity_4.g",
        "O": "./valence_gaussians/O_pseudodensity_4.g",
        "Cu": "./valence_gaussians/Cu_pseudodensity_4.g",
    },
    "cutoff": 12,
}


elements = ["Cu", "C", "O"]
config = {
    "model": {
        "name": "singlenn",
        "get_forces": False,
        "num_layers": 3,
        "num_nodes": 20,
    },
    "optim": {
        "device": "cpu",
        "force_coefficient": 0.0,
        "lr": 1e-2,
        "batch_size": 10,
        "epochs": 100,
    },
    "dataset": {
        "raw_data": images,
        "val_split": 0,
        "elements": elements,
        "fp_scheme": "gmpordernorm",
        "fp_params": GMP,
        "save_fps": True,
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        "logger": False,
    },
}

trainer = AtomsTrainer(config)
trainer.train()

predictions = trainer.predict(images[:10])

true_energies = np.array([image.get_potential_energy() for image in images])
pred_energies = np.array(predictions["energy"])

print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))
