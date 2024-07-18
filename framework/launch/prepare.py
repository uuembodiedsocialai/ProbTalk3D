import os
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from framework.tools.runid import generate_id
import hydra
import torch


# Local paths
def code_path(path=""):
    code_dir = hydra.utils.get_original_cwd()
    code_dir = Path(code_dir)
    return str(code_dir / path)


def working_path(path):
    return str(Path(os.getcwd()) / path)


# fix the id for this run
ID = generate_id()


def generate_id():
    return ID


def get_last_checkpoint(path, version, ckpt_name="last.ckpt"):
    output_dir = Path(hydra.utils.to_absolute_path(path))
    last_ckpt_path = output_dir / f"lightning_logs/version_{version}/checkpoints" / ckpt_name
    return str(last_ckpt_path)


OmegaConf.register_new_resolver("code_path", code_path)
OmegaConf.register_new_resolver("working_path", working_path)
OmegaConf.register_new_resolver("generate_id", generate_id)
OmegaConf.register_new_resolver("absolute_path", hydra.utils.to_absolute_path)
OmegaConf.register_new_resolver("get_last_checkpoint", get_last_checkpoint)


# Remove some warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

warnings.filterwarnings(
    "ignore", ".*pyprof will be removed by the end of June.*"
)

warnings.filterwarnings(
    "ignore", ".*pandas.Int64Index is deprecated.*"
)

warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck*"
)

warnings.filterwarnings(
    "ignore", ".*Our suggested max number of worker in current system is*"
)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# set to medium or high
torch.set_float32_matmul_precision('medium')
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()