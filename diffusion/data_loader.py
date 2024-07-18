import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
import sys


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train" or self.data_type == "val" or self.data_type == "test":
            subject = file_name.split("_")[0]
            id_one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
            id_one_hot = torch.tensor(id_one_hot)
            emotion_idx = int(file_name.split("_")[2])
            emotion_one_hot = torch.eye(8)[emotion_idx]
            intensity_idx = int(file_name.split("_")[3][:1])
            intensity_one_hot = torch.eye(3)[intensity_idx]
            one_hot = torch.cat([id_one_hot, emotion_one_hot, intensity_one_hot], dim=0).float()
        else:
            one_hot = self.one_hot_labels

        return torch.FloatTensor(audio), vertice, torch.FloatTensor(template), torch.FloatTensor(
            one_hot), file_name

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    sys.stdout.flush()
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_path, args.dataset, args.wav_path)
    vertices_path = os.path.join(args.data_path, args.dataset, args.vertices_path)

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/hubert-xlarge-ls960-ft")  # HuBERT uses the processor of Wav2Vec 2.0

    template_file = os.path.join(args.data_path, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    all_subjects = args.test_subjects.split()
    for r, ds, fs in os.walk(vertices_path):
        for f in tqdm(fs, disable=True):
            if f.endswith("npy"):
                m_path = os.path.join(r, f)
                key = f.replace("npy", "wav")

                # get sample info from the name and add it to the dict for the splits
                sentence_id = int(key.split("_")[1])
                subject_id = key.split("_")[0]

                # skip subjects not included in the training or test sets for faster loading
                if subject_id not in all_subjects:
                    continue

                data[key]["vertice"] = m_path
                temp = templates.get(subject_id, np.zeros(args.vertice_dim))    # [5023, 3]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1))                      # [15069]

                wav_path = os.path.join(audio_path, f.replace("npy", "wav"))
                if not os.path.exists(wav_path):
                    del data[key]
                    print("Audio Data Not Found! ", wav_path)
                else:
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(processor(speech_array, return_tensors="pt", padding="longest",
                                                        sampling_rate=sampling_rate).input_values)
                    data[key]["audio"] = input_values

    data_neutral = args.dataset + "_n"
    data_emotional = args.dataset + "_e"
    splits = {
        data_neutral: {
            'train': range(1, 33),
            'val': range(33, 37),
            'test': range(37, 41)
        },  # neutral
        data_emotional: {
            'train': range(1, 25),
            'val': range(25, 28),
            'test': range(28, 31)
        },  # emotional
    }

    emotions = {
        'neutral': [0],
        'emotional': [1, 2, 3, 4, 5, 6, 7]
    }

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    for k, v in data.items():
        subject_id = k.split("_")[0]
        sentence_id = int(k.split("_")[1])
        emotions_id = int(k.split("_")[2])
        if emotions_id in emotions['neutral']:
            idx = args.dataset + '_n'
        elif emotions_id in emotions['emotional']:
            idx = args.dataset + '_e'
        else:
            raise ValueError("Unknown emotion")

        if subject_id in subjects_dict["train"] and sentence_id in splits[idx]['train']:
            train_data.append(v)
        elif subject_id in subjects_dict["val"] and sentence_id in splits[idx]['val']:
            valid_data.append(v)
        elif subject_id in subjects_dict["test"] and sentence_id in splits[idx]['test']:
            test_data.append(v)
        else:
            raise ValueError("Unknown split for input:", k)

    print("number of training data:", len(train_data),
          "validation data:", len(valid_data),
          "test data:", len(test_data))
    sys.stdout.flush()
    return train_data, valid_data, test_data, subjects_dict


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(args):
    g = torch.Generator()
    g.manual_seed(0)
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker,
                                       generator=g)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset


if __name__ == "__main__":
    get_dataloaders()

