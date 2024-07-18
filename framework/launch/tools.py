from omegaconf import DictConfig
import os


def resolve_cfg_path(cfg: DictConfig):
    working_dir = os.getcwd()
    cfg.working_dir = working_dir
