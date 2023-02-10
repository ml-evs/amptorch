from dataclasses import dataclass, field
import torch.nn as nn
import torch
from typing import Optional, Union, List
from amptorch.model import CustomLoss


@dataclass
class ModelConfig:
    num_layers: int = 5
    """No. of hidden layers"""

    num_nodes: int = 20
    """No. of nodes per layer"""

    get_forces: bool = True
    """Compute per-atom forces"""

    batchnorm: bool = False
    """Enables batch normalization"""

    activation: nn.Module = nn.Tanh
    """Activation function"""


@dataclass
class OptimizerConfig:
    gpus: int = 0
    """No. of gpus to use, 0 for cpu"""

    force_coefficient: float = 0
    """If force training, coefficient to weight the force component by"""

    lr: float = 0.1
    """Initial learning rate"""

    batch_size: int = 32
    """Batch size"""

    epochs: int = 100
    """Max training epochs"""

    optimizer: torch.optim.Optimizer = torch.optim.Adam
    """Training optimizer"""

    loss_fn: nn.Module = CustomLoss
    """Loss function to optimize"""

    loss: str = "mse"
    """Control loss function criterion, "mse" or \"mae\""""

    metric: str = "mae"
    """Metrics to be reported by, "mse" or \"mae\""""

    cp_metric: str = "energy"
    """Property based on which the model is saved. "energy" or \"forces\""""

    scheduler: Optional[dict] = None
    """Learning rate scheduler to use, 
    e.g., `{"policy": "StepLR", "params": {"step_size": 10, "gamma": 0.1}`
    """


@dataclass
class DatasetConfig:
    raw_data: Union[str, list] = None
    """Path to ASE trajectory or database or list of Atoms objects"""

    lmdb_path: str = None
    """Path to LMDB database file for dataset too large to fit in memory
    Specify either "raw_data" or "lmdb_path"
    LMDB construction can be found in examples/construct_lmdb.py

    """

    val_split: float = 0
    """Proportion of training set to use for validation"""

    elements: List[str] = None
    """List of unique elements in dataset, optional (default: computes unique elements)"""

    fp_scheme: str = "gaussian"
    """Fingerprinting scheme to feature dataset, "gaussian" or "gmp" (default: "gaussian")"""

    fp_params: dict = None
    """Fingerprint parameters, see examples for correct layout"""

    cutoff_params: dict = None
    """Cutoff function - polynomial or cosine,
    Polynomial - {"cutoff_func": "Polynomial", "gamma": 2.0}
    Cosine     - {"cutoff_func": "Cosine"}
    """

    save_fps: bool = True
    """Write calculated fingerprints to disk"""

    scaling: dict = None
    """Feature scaling scheme, normalization or standardization
    - normalization (scales features between "range")
        - {"type": "normalize", "range": (0, 1)}
    - standardization (scales data to mean=0, stdev=1)
    """


@dataclass
class CommandConfig:
    debug: bool = False
    """Debug mode, does not write/save checkpoints/results"""

    dtype: object = (torch.DoubleTensor,)
    """Pytorch level of precision"""

    run_dir: str = "./"
    """Path to run trainer, where logs are to be saved"""

    seed: int = 0
    """Random seed"""

    identifier: str = (None,)
    """Unique identifer to experiment, optional"""

    verbose: bool = True
    """Print training scores"""

    logger: bool = False
    """Log results to Weights and Biases (https://www.wandb.com/)
  wandb offers a very clean and flexible interface to monitor results online.
  A free account is necessary to view and log results.
  """


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cmd: CommandConfig = field(default_factory=CommandConfig)
