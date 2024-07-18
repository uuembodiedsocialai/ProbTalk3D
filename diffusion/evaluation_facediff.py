import logging
import os
from pathlib import Path
import pickle
import torch
import numpy as np
import argparse
import datetime
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from models import FaceDiff
from data_loader import get_dataloaders
from utils import *

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
       "0": "low",
       "1": "medium",
       "2": "high",
   }


def lve_compute(vertices_gt, vertices_pred, mouth_map):
    # L2_dis_mouth_max: (428, T, 3), 428 vertex indices
    vertices_gt = np.array(vertices_gt)
    vertices_pred = np.array(vertices_pred)
    L2_dis_mouth_max = np.array([np.square(vertices_gt[:, v, :] - vertices_pred[:, v, :]) for v in mouth_map])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))    # (T, 428, 3)
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)
    return L2_dis_mouth_max


def seq_std_compute(motion, map):
    # map: 1501 vertex indices
    # motion[:, v, :]: (T, 3)
    L2_dis = np.array([np.square(motion[:, v, :]) for v in map])    # (1501, T, 3)
    L2_dis = np.transpose(L2_dis, (1, 0, 2))                        # (T, 1501, 3)
    L2_dis = np.sum(L2_dis, axis=2)
    L2_dis = np.std(L2_dis, axis=0)
    std = np.mean(L2_dis)
    return std


@torch.no_grad()
def test_diff(args, model, test_loader, epoch, diffusion, device="cuda:0"):
    result_path = os.path.join(args.result_path)
    os.makedirs(result_path, exist_ok=True)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    model_path = f'{args.save_path}/{args.model}_{args.dataset}_{epoch}.pth'
    print("load model from...", model_path)
    sys.stdout.flush()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Device checking:", device)

    # load templates for evaluation
    with open(Path(f"{parent_dir}/datasets/regions/lve.txt"))as f:
        maps = f.read().split(",")
        mouth_map = [int(i) for i in maps]
    with open(Path(f"{parent_dir}/datasets/regions/fdd.txt")) as f:
        maps = f.read().split(",")
        upper_map = [int(i) for i in maps]
    with open(Path(f"{parent_dir}/datasets/templates_mead_vert.pkl"), 'rb') as f:
        templates = pickle.load(f, encoding='latin1')   # [5023, 3]

    sr = 16000
    # count frame numbers
    seq_count = 0
    frame_count = 0
    vertices_all_gt = []        # mve, lve: gt
    vertices_all_pred = []      # mve, lve: first prediction
    mee_all = []                # mean value
    ce_all = []                 # closest value
    motion_std_difference = []  # fdd: first prediction
    diversity = 0               # 2 subsets
    for idx, (audio, vertice, template, one_hot_all, file_name) in enumerate(test_loader):
        vertice = vertice_path = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertices_npy_gt = vertice.copy()                    # (T, 5023, 3)

        vertice_path = os.path.split(vertice_path)[-1][:-4]
        emo = emo_dict[str(vertice_path.split("_")[2])]     # retrieve emotion
        ints = int_dict[str(vertice_path.split("_")[3])]    # retrieve intensity
        gt_path = os.path.join(args.result_path, f"{vertice_path}_{emo}_{ints}_gt.npy")

        if (idx + 1) % 100 == 0:
            print(f"Saving gt: {gt_path}")
            sys.stdout.flush()
            np.save(gt_path, vertices_npy_gt)

        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)
        vertice = vertice.reshape(1, vertice.shape[0], vertice.shape[1] * vertice.shape[2])

        audio, vertice = audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

        num_frames = int(audio.shape[-1] / sr * args.output_fps)
        shape = (1, num_frames - 1, args.vertice_dim) if num_frames < vertice.shape[1] else vertice.shape

        train_subject = file_name[0].split("_")[0]
        sys.stdout.flush()
        assert train_subject in train_subjects_list, "train_subject not in train_subjects_list"

        one_hot = one_hot_all.to(device=device)
        ce_lve_set = []  # save 10 lve values
        motion_set = []  # save 10 samples
        for sample_idx in range(1, args.num_samples + 1):
            # use ddim
            sample = diffusion.ddim_sample_loop(
                model,
                shape,
                clip_denoised=False,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                },
                skip_timesteps=args.skip_steps,     # skip 900 timesteps
                init_image=None,
                progress=None,
                dump_steps=None,
                noise=None,
                const_noise=False,
                device=args.device,
            )
            sample = sample.squeeze()
            sample = sample.detach().cpu().numpy()  # (T, 5023*3)

            vertices_npy_pred = sample.reshape(-1, 5023, 3)
            vertices_npy_pred = vertices_npy_pred[:vertices_npy_gt.shape[0], :, :]  # (T, 5023, 3)
            vertices_npy_gt = vertices_npy_gt[:vertices_npy_pred.shape[0], :, :]    # (T, 5023, 3)

            """CE: compute lve for each samples"""
            ce_lve = lve_compute(vertices_gt=list(vertices_npy_gt),
                                 vertices_pred=list(vertices_npy_pred),
                                 mouth_map=mouth_map)
            ce_lve_set.append(ce_lve)               # (T,)

            # save 10 samples
            motion_set.append(vertices_npy_pred)    # (T, 5023, 3)
            torch.cuda.empty_cache()

            if args.num_samples != 1:
                out_path = f"{vertice_path}_{emo}_{ints}_{sample_idx}.npy"
            else:
                out_path = f"{vertice_path}_{emo}_{ints}_one.npy"

            # save vertices (a small part)
            if (idx + 1) % 100 == 0:                # save every 100 keyids for checking
                if (sample_idx + 1) % 3 == 1:       # save 3 samples in 10
                    print(f"Saving pred: {out_path}")
                    np.save(os.path.join(args.result_path, out_path), sample)

        """MVE, LVE: save prediction of all audio samples"""
        vertices_all_gt.extend(list(vertices_npy_gt))               # length T of items (5023, 3)
        vertices_all_pred.extend(list(motion_set[0]))               # use the first sample

        """MEE: mean over 10 samples"""
        motion_set_stack = np.stack(motion_set, axis=0)
        vertices_npy_pred_mean = np.mean(motion_set_stack, axis=0)  # (T, 5023, 3)
        mee_lve = lve_compute(vertices_gt=list(vertices_npy_gt),
                              vertices_pred=list(vertices_npy_pred_mean),
                              mouth_map=mouth_map)
        mee_all.extend(list(mee_lve))

        """CE: closest lve in 10 samples"""
        smallest_lve = None
        smallest_lve_value = float('inf')                           # start with an infinitely large value
        for lve_of_one_seq in ce_lve_set:
            lve_value = np.sum(lve_of_one_seq)
            if lve_value < smallest_lve_value:
                smallest_lve_value = lve_value
                smallest_lve = lve_of_one_seq                       # (T,)
        assert smallest_lve is not None, "No smallest distance found"
        ce_all.extend(list(smallest_lve))

        # count sequence, frame numbers
        frame_count += vertices_npy_gt.shape[0]
        seq_count += 1

        """FDD computation: use the first sample"""
        subject_template = templates[train_subject].reshape(1, 5023, 3)
        subject_template = subject_template.detach().cpu().numpy()  # (1, 5023, 3)
        upper_std_gt = seq_std_compute(motion=vertices_npy_gt - subject_template,
                                       map=upper_map)
        upper_std_pred = seq_std_compute(motion=motion_set[0] - subject_template,
                                         map=upper_map)
        motion_std_difference.append(upper_std_gt - upper_std_pred)

        """Diversity computation"""
        np.random.shuffle(motion_set)                               # list of (T, 5023, 3) number=10
        subset1 = motion_set[:5]
        subset2 = motion_set[5:]
        motion_diversity = 0
        for sample1, sample2 in zip(subset1, subset2):
            motion_diversity += np.linalg.norm(sample1 - sample2, axis=2).mean(axis=1).mean()
        if len(subset1) == 5 and len(subset2) == 5:
            motion_diversity /= len(subset1)
            diversity += motion_diversity
        else:
            raise ValueError("Subset length mismatch 5")

        part_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{part_time} Done sampling: {vertice_path} ")
        sys.stdout.flush()
        torch.cuda.empty_cache()

    print('Total sequence number: {}'.format(seq_count))
    print('Total frame number: {}'.format(frame_count))
    sys.stdout.flush()
    if seq_count == 0:
        print("No sequences were processed. Unable to compute metrics.")
    else:
        """MVE computation"""
        vertices_all_gt = np.array(vertices_all_gt)  # (frame_cunt, 5023, 3)
        vertices_all_pred = np.array(vertices_all_pred)
        vertices_dis = np.linalg.norm(vertices_all_gt - vertices_all_pred, axis=2)
        print('MVE: {:.4e}'.format(np.mean(vertices_dis)))

        """LVE computation"""
        L2_dis_mouth_max = lve_compute(vertices_gt=vertices_all_gt,
                                       vertices_pred=vertices_all_pred,
                                       mouth_map=mouth_map)
        print('LVE: {:.4e}'.format(np.mean(L2_dis_mouth_max)))

        """MEE computation"""
        print('MEE: {:.4e}'.format(np.mean(mee_all)))

        """CE computation"""
        print('CE: {:.4e}'.format(np.mean(ce_all)))

        """FDD computation"""
        print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))

        """Divertiy computation"""
        print('Diversity: {:.4e}'.format(diversity / seq_count))

        print(f"All the sampling are done. ")
        sys.stdout.flush()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="mead", help='Name of the dataset folder. eg: BIWI')
    parser.add_argument("--data_path", type=str, default=f"{parent_dir}/datasets/")
    parser.add_argument("--vertice_dim", type=int, default=15069, help='number of vertices - 23370*3 for BIWI dataset')
    parser.add_argument("--feature_dim", type=int, default=256, help='Latent Dimension to encode the inputs to')
    parser.add_argument("--gru_dim", type=int, default=256, help='GRU Vertex decoder hidden size')
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertex", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=50, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="face_diffuser", help='name of the trained model')
    parser.add_argument("--template_file", type=str, default="templates_mead_vert.pkl",
                        help='path of the train subject templates')
    parser.add_argument("--save_path", type=str, default="outputs/model", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="results/evaluation", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="M003 M005 M007 M009 M011 M012 M013 M019 "
                                                              "M022 M023 M024 M025 M026 M027 M028 M029 "
                                                              "M030 M031 W009 W011 W014 W015 W016 W018 "
                                                              "W019 W021 W023 W024 W025 W026 W028 W029")
    parser.add_argument("--val_subjects", type=str, default="M003 M005 M007 M009 M011 M012 M013 M019 "
                                                            "M022 M023 M024 M025 M026 M027 M028 M029 "
                                                            "M030 M031 W009 W011 W014 W015 W016 W018 "
                                                            "W019 W021 W023 W024 W025 W026 W028 W029")
    parser.add_argument("--test_subjects", type=str, default="M003 M005 M007 M009 M011 M012 M013 M019 "
                                                             "M022 M023 M024 M025 M026 M027 M028 M029 "
                                                             "M030 M031 W009 W011 W014 W015 W016 W018 "
                                                             "W019 W021 W023 W024 W025 W026 W028 W029")
    parser.add_argument("--input_fps", type=int, default=50,
                        help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=25,
                        help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--diff_steps", type=int, default=1000, help='number of diffusion steps')
    parser.add_argument("--skip_steps", type=int, default=900, help='number of diffusion steps to skip during inference')
    parser.add_argument("--num_samples", type=int, default=10, help='number of samples to generate per audio')
    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the start time
    print(f"Start time: {start_time}")
    sys.stdout.flush()
    assert torch.cuda.is_available()
    diffusion = create_gaussian_diffusion(args)

    model = FaceDiff(
        args,
        vertice_dim=args.vertice_dim,
        latent_dim=args.feature_dim,
        diffusion_steps=args.diff_steps,
        gru_latent_dim=args.gru_dim,
        num_layers=args.gru_layers,
    )
    print("model parameters: ", count_parameters(model))
    sys.stdout.flush()
    cuda = torch.device(args.device)
    dataset = get_dataloaders(args)

    test_diff(args, model, dataset["test"], args.max_epoch, diffusion, device=args.device)

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"End time: {end_time}")

