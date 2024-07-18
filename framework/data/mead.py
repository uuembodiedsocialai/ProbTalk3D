import logging
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from rich.progress import track

from .base import BaseDataModule
from .utils import get_split_keyids
from framework.data.tools.collate import audio_normalize
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class MeadDataModule(BaseDataModule):
    def __init__(self, batch_size: int,
                 num_workers: int,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.save_hyperparameters(logger=False)

        self.split_path = self.hparams.split_path
        self.one_hot_dim = self.hparams.one_hot_dim
        self.Dataset = MEAD
        # Get additional info of the dataset
        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats


class MEAD(Dataset):
    def __init__(self, data_name: str,
                 motion_path: str,
                 audio_path: str,
                 split_path: str,
                 load_audio: str,
                 split: str,
                 tiny: bool,
                 progress_bar: bool,
                 debug: bool,
                 **kwargs):
        super().__init__()
        self.data_name = data_name
        self.load_audio = load_audio

        ids = get_split_keyids(path=split_path, split=split)
        if progress_bar:
            enumerator = enumerate(track(ids, f"Loading {data_name} {split} dataset"))
        else:
            enumerator = enumerate(ids)

        if tiny:
            max_data = 2
        elif not tiny and debug:
            max_data = 8
        else:
            max_data = np.inf

        motion_data_all = {}
        shape_data_all = {}
        audio_data_all = {}

        # Initializing MinMaxScalars for exp and jaw
        exp_scaler = MinMaxScaler()
        exp_sample = [-3, 3]        # accepted exp value range
        a = np.array(exp_sample)[:, np.newaxis]
        a = np.repeat(a, 50, axis=1)
        exp_scaler.fit(a)           # scales data to [0,1] range

        jaw_scaler = MinMaxScaler()
        jaw_sample = [-0.1, 0.5]    # accepted jaw values range
        b = np.array(jaw_sample)[:, np.newaxis]
        b = np.repeat(b, 3, axis=1)
        jaw_scaler.fit(b)           # scales data to [0,1] range

        if load_audio:
            for i, id in enumerator:
                if len(motion_data_all) >= max_data:
                    break
                # load 3DMEAD dataset
                key, motion_data, shape_data, audio_data = load_data(keyid=id,
                                                                     motion_path=Path(motion_path),
                                                                     audio_path=Path(audio_path),
                                                                     max_data=max_data,
                                                                     load_audio=load_audio,
                                                                     exp_scaler=exp_scaler,
                                                                     jaw_scaler=jaw_scaler,
                                                                     split=split)
                motion_data_all.update(dict(zip(key, motion_data)))
                shape_data_all.update(dict(zip(key, shape_data)))
                audio_data_all.update(dict(zip(key, audio_data)))
        else:
            for i, id in enumerator:
                if len(motion_data_all) >= max_data:
                    break
                # load 3DMEAD dataset
                key, motion_data, shape_data = load_data(keyid=id,
                                                         motion_path=Path(motion_path),
                                                         audio_path=Path(audio_path),
                                                         max_data=max_data,
                                                         load_audio=load_audio,
                                                         exp_scaler=exp_scaler,
                                                         jaw_scaler=jaw_scaler,
                                                         split=None)
                motion_data_all.update(dict(zip(key, motion_data)))
                shape_data_all.update(dict(zip(key, shape_data)))

        self.motion_data = motion_data_all
        self.shape_data = shape_data_all
        if load_audio:
            self.audio_data = audio_data_all
        self.keyids = list(motion_data_all.keys())  # file name
        self.nfeats = self[0]["motion"].shape[2]    # number of feature
        print(f"The number of loaded data pair is: {len(self.motion_data)}")
        print(f"Number of features of a motion frame: {self.nfeats}")

    def load_keyid(self, keyid):
        if self.load_audio:
            element = {"motion": self.motion_data[keyid], "shape": self.shape_data[keyid], "audio": self.audio_data[keyid],
                       "keyid": keyid}
        else:
            element = {"motion": self.motion_data[keyid], "shape": self.shape_data[keyid], "keyid": keyid}
        return element

    def __getitem__(self, index):
        keyid = self.keyids[index]
        element = self.load_keyid(keyid)
        return element

    def __len__(self):
        return len(self.keyids)

    def __repr__(self):
        return f"{self.data_name} dataset: ({len(self)}, _, ..)"


def load_data(keyid, motion_path, audio_path, max_data, load_audio, exp_scaler, jaw_scaler, split):
    try:
        motion_dir = list(motion_path.glob(f"{keyid}*.npy"))
        motion_key = [directory.stem for directory in motion_dir]
        audio_dir = list(audio_path.glob(f"{keyid}*.wav"))
        audio_key = [directory.stem for directory in audio_dir]
    except FileNotFoundError:
        return None

    keys = []
    motion_data = []
    shape_data = []
    audio_data = []
    if load_audio:  # second stage
        for key in motion_key:
            if len(keys) >= max_data:
                keys = keys[:max_data]
                motion_data = motion_data[:max_data]
                shape_data = shape_data[:max_data]
                audio_data = audio_data[:max_data]
                break

            if key in audio_key:
                key_split = key.split("_")
                load_key = None
                if split == "train":
                    if int(key_split[2]) == 0:
                        if int(key_split[1]) in range(33):  # load sentence 1-32 (neutral)
                            load_key = key
                            keys.append(key)
                    else:
                        if int(key_split[1]) in range(25):  # load sentence 1-24 (emotional)
                            load_key = key
                            keys.append(key)
                elif split == "val":
                    if int(key_split[2]) == 0:
                        if int(key_split[1]) in range(33, 37):  # load sentence 33-36 (neutral)
                            load_key = key
                            keys.append(key)
                    else:
                        if int(key_split[1]) in range(25, 28):  # load sentence 25-27 (emotional)
                            load_key = key
                            keys.append(key)
                elif split == "test":
                    if int(key_split[2]) == 0:
                        if int(key_split[1]) in range(37, 41):  # load sentence 37-40 (neutral)
                            load_key = key
                            keys.append(key)
                    else:
                        if int(key_split[1]) in range(28, 31):  # load sentence 28-30 (emotional)
                            load_key = key
                            keys.append(key)

                if load_key is not None:
                    m_index = motion_key.index(load_key)
                    m_dir = motion_dir[m_index]
                    m_npy = np.load(m_dir)

                    # save motion data
                    exp = np.squeeze(m_npy[:, :, 300:350])
                    jaw = np.squeeze(m_npy[:, :, 400:])
                    normalized_exp = exp_scaler.transform(exp)
                    normalized_jaw = jaw_scaler.transform(jaw)
                    normalized_exp_jaw = np.concatenate((normalized_exp, normalized_jaw), axis=1)
                    motion_data.append(torch.from_numpy(normalized_exp_jaw).unsqueeze(0))
                    # save shape data
                    shape_data.append(torch.from_numpy(m_npy[:, :, :300]))
                    # save audio data
                    a_index = audio_key.index(key)
                    a_dir = audio_dir[a_index]

                    speech_array, _ = librosa.load(a_dir, sr=16000)
                    speech_array = audio_normalize(speech_array)
                    audio_data.append(speech_array)
                else:
                    # print(f"Pass {key}")
                    pass
            else:
                print(f"No matching audio file for {key}")
                pass

        return keys, motion_data, shape_data, audio_data
    else:     # first stage
        for dir in motion_dir:
            if len(keys) >= max_data:
                keys = keys[:max_data]
                motion_data = motion_data[:max_data]
                shape_data = shape_data[:max_data]
                break

            m_npy = np.load(dir)
            # save file name as key
            keys.append(dir.stem)
            # save motion data
            exp = np.squeeze(m_npy[:, :, 300:350])
            jaw = np.squeeze(m_npy[:, :, 400:])
            normalized_exp = exp_scaler.transform(exp)
            normalized_jaw = jaw_scaler.transform(jaw)
            normalized_exp_jaw = np.concatenate((normalized_exp, normalized_jaw), axis=1)
            motion_data.append(torch.from_numpy(normalized_exp_jaw).unsqueeze(0))
            # save shape data
            shape_data.append(torch.from_numpy(m_npy[:, :, :300]))
        return keys, motion_data, shape_data





