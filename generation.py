import logging
import os
import torch
import pickle
import numpy as np
import warnings
from natsort import os_sorted

import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import framework.launch.prepare  # noqa
from framework.model.utils.tools import load_checkpoint, detach_to_numpy
from framework.data.utils import get_split_keyids

logger = logging.getLogger(__name__)

emo_dict = {
       "0": "neutral",   # only have one intensity level
       "1": "happy",
       "2": "sad",
       "3": "surprised",
       "4": "fear",
       "5": "disgusted",
       "6": "angry",
       "7": "contempt"
   }

int_dict = {
       "0": "low",   # only have one intensity level
       "1": "medium",
       "2": "high",
   }


@hydra.main(version_base=None, config_path="configs", config_name="generation")
def _sample(cfg: DictConfig):
    return sample(cfg)


# generate one or multiple samples
def cfg_mean_nsamples_resolution(cfg):
    # If VAE take mean value, set number_of_samples=1
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1


# set VAE variant output path
def get_path_vae(sample_path: Path, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "none" if fact == 1 else f"{fact}"
    path = sample_path / f"{fact_str}{extra_str}"
    return path


# set VQVAE variant output path
def get_path_vqvae(sample_path: Path, onesample: bool, temperature: float, k: float):
    extra_str = "" if onesample else "_multi"
    tem_str = f"{temperature}"
    k_str = "" if k == 1 else f"_{k}"
    path = sample_path / f"{tem_str}{k_str}{extra_str}"
    return path


# prediction
def sample(newcfg: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load previous configs
    prevcfg = OmegaConf.load(Path(newcfg.folder) / ".hydra/config.yaml")
    # Merge configs to overload them
    cfg = OmegaConf.merge(prevcfg, newcfg)

    onesample = cfg_mean_nsamples_resolution(cfg)

    logger.info("Sample script. The outputs will be stored in:")
    folder_name = cfg.folder.split("/")[-1]
    output_dir = Path(cfg.path.code_dir) / f"results/generation/{cfg.experiment}/{folder_name}"
    path = None
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        path = get_path_vae(output_dir, onesample, cfg.mean, cfg.fact)
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        if not cfg.sample:
            path = get_path_vqvae(output_dir, onesample, "none", cfg.k)
        else:
            path = get_path_vqvae(output_dir, onesample, cfg.temperature, cfg.k)
    if path is None:
        raise ValueError("No model specified in the config file.")
    else:
        path.mkdir(exist_ok=True, parents=True)
        logger.info(f"{path}")

    # update the motion prior if needed
    if cfg.folder_prior is not None and cfg.version_prior is not None:
        if os.path.exists(cfg.folder_prior):
            OmegaConf.update(cfg, "model.folder_prior", cfg.folder_prior)
            OmegaConf.update(cfg, "model.version_prior", cfg.version_prior)
        else:
            logger.info(f"Using default motion prior.")

    # save config to check
    OmegaConf.save(cfg, output_dir / "merged_config.yaml")

    from hydra.utils import instantiate
    logger.info("Loading model")
    last_ckpt_path = cfg.last_ckpt_path
    model = instantiate(cfg.model,
                        nfeats=cfg.nfeats,
                        split_path=cfg.data.split_path,
                        one_hot_dim=cfg.data.one_hot_dim,
                        resumed_training=False,
                        logger_name="none",
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    # move model to cuda
    if cfg.device is None:
        device_index = cfg.trainer.devices[0]
    else:
        device_index = cfg.device
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if device_index < num_devices:
            model.to(f"cuda:{device_index}")
        else:
            model.to(f"cuda:0")
    print("device checking:", model.device)

    load_checkpoint(model, last_ckpt_path, eval_mode=True, device=model.device)
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        model.motion_prior.sample_mean = cfg.mean
        model.motion_prior.fact = cfg.fact
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        model.temperature = cfg.temperature
        model.k = cfg.k

    from rich.progress import Progress
    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)

    # load audio
    from framework.data.tools.collate import audio_normalize
    import librosa
    audio_dir = Path(cfg.input_path)
    files = list(audio_dir.glob("*.wav"))
    files = os_sorted(files)
    keys = [file.stem for file in files]

    # set style one hot
    ids = get_split_keyids(path=cfg.data.split_path, split="train")
    emotions = list(range(8))
    intensitys = list(range(3))
    import random

    # load audio
    name = []
    audio_data = []
    key_id = []
    for idx, key in enumerate(keys):
        name.append(key)

        file = files[idx]
        speech_array, _ = librosa.load(file, sr=16000)
        speech_array = audio_normalize(speech_array)
        audio_data.append(speech_array)

        # setting keyid for style control
        if cfg.id is not None:
            if idx > len(cfg.id)-1:             # if did not set the keyid
                print(f"choosing a random id for audio '{key}'")
                id = random.choice(ids)
            else:
                if cfg.id[idx] in ids:
                    id = cfg.id[idx]
                else:
                    print(f"id {cfg.id[idx]} is not supported, choosing a random one")
                    id = random.choice(ids)
        else:
            print(f"choosing a random id for audio '{key}'")
            id = random.choice(ids)
        if cfg.emotion is not None:
            if idx > len(cfg.emotion)-1:        # if did not set the emotion
                print(f"choosing a random emotion for audio '{key}'")
                emotion = random.choice(emotions)
            else:
                emotion = next((k for k, v in emo_dict.items() if v == cfg.emotion[idx]), None)
                if emotion is None:
                    print(f"emotion {cfg.emotion[idx]} is not supported, choosing a random one")
                    emotion = random.choice(emotions)
        else:
            print(f"choosing a random emotion for audio '{key}'")
            emotion = random.choice(emotions)
        if cfg.intensity is not None:
            if idx > len(cfg.intensity)-1:      # if did not set the intensity
                print(f"choosing a random intensity for audio '{key}'")
                intensity = random.choice(intensitys)
            else:
                if cfg.intensity[idx] in intensitys:
                    intensity = cfg.intensity[idx]
                else:
                    print(f"intensity {cfg.intensity[idx]} is not supported, choosing a random one")
                    intensity = random.choice(intensitys)
        else:
            print(f"choosing a random intensity for audio '{key}'")
            intensity = random.choice(intensitys)
        keyid = '{}_x_{}_{}'.format(id, emotion, intensity)
        key_id.append(keyid)

    with open("datasets/scaler_exp.pkl", 'rb') as f:
        scaler_exp = pickle.load(f)
    with open("datasets/scaler_jaw.pkl", 'rb') as f:
        scaler_jaw = pickle.load(f)

    npypath = None
    with torch.no_grad():
        with Progress(transient=True) as progress:
            for i in range(len(name)):
                task = progress.add_task("Generating", total=len(name[i]))
                progress.update(task, description=f"Sampling {name[i]}...")
                for index in range(cfg.number_of_samples):
                    batch = {"audio": [audio_data[i]],
                             "keyid": [key_id[i]]}
                    print("Audio:", name[i], "Style:", batch["keyid"], "Sample number:", index+1)

                    exp_jaw_pred = model(batch, sample=cfg.sample)              # [1, T, 53]
                    exp_jaw_pred = detach_to_numpy(exp_jaw_pred.squeeze(0))     # (T, 53)

                    # denormalization pred
                    shape_prefix = np.zeros((exp_jaw_pred.shape[0], 300))
                    exp_suffix = np.zeros((exp_jaw_pred.shape[0], 50))

                    inverse_exp_pred = scaler_exp.inverse_transform(exp_jaw_pred[:, :50])
                    inverse_jaw_pred = scaler_jaw.inverse_transform(exp_jaw_pred[:, 50:])
                    inverse_exp_pred = np.concatenate((inverse_exp_pred, exp_suffix), axis=1)               # (T, 100)
                    inverse_exp_jaw_pred = np.concatenate((inverse_exp_pred, inverse_jaw_pred), axis=1)     # (T, 103)
                    seq_pred = np.concatenate((shape_prefix, inverse_exp_jaw_pred), axis=1)                 # (T, 403)
                    seq_pred = np.expand_dims(seq_pred, axis=0)     # (1, T, 403)

                    save_keyid = key_id[i].split("_")
                    emo = emo_dict[str(save_keyid[2])]
                    ints = int_dict[str(save_keyid[3])]
                    if cfg.number_of_samples > 1:
                        npypath = path / f"{name[i]}_{save_keyid[0]}_{emo}_{ints}({index+1}).npy"
                    else:
                        npypath = path / f"{name[i]}_{save_keyid[0]}_{emo}_{ints}.npy"

                    np.save(npypath, seq_pred)
                progress.update(task, advance=1)

    if npypath is not None:
        logger.info(f"All the sampling are done. You can find them here:\n{npypath.parent}")
    else:
        logger.error("No audio input found.")


if __name__ == '__main__':
    _sample()
