import argparse
import os
import torch
import numpy as np

from data_loader import get_dataloaders
from diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm
import datetime
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from models import FaceDiff
from utils import *


def trainer_diff(args, train_loader, dev_loader, model, diffusion, optimizer, epoch, device="cuda:0"):
    train_losses = []
    val_losses = []

    save_path = os.path.join(args.save_path)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    iteration = 0

    for e in range(epoch + 1):
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the start time
        print(f"Epoch {e + 1} start time: {start_time}")
        sys.stdout.flush()
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=True)
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)

            vertice = vertice.reshape(1, vertice.shape[0], vertice.shape[1]*vertice.shape[2])
            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot = template.to(device=device), one_hot.to(device=device)
            loss = diffusion.training_losses(
                model,
                x_start=vertice,
                t=t,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                }
            )['loss']

            loss = torch.mean(loss)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                del audio, vertice, template, one_hot
                torch.cuda.empty_cache()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log)))
            print("(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log)))
            sys.stdout.flush()

        train_losses.append(np.mean(loss_log))

        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            # to gpu
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)

            vertice = vertice.reshape(1, vertice.shape[0], vertice.shape[1] * vertice.shape[2])
            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)
            train_subject = file_name[0].split("_")[0]
            if train_subject in train_subjects_list:
                one_hot = one_hot_all
                loss = diffusion.training_losses(
                    model,
                    x_start=vertice,
                    t=t,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    }
                )['loss']

                loss = torch.mean(loss)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    one_hot = one_hot_all[:, iter, :]
                    loss = diffusion.training_losses(
                        model,
                        x_start=vertice,
                        t=t,
                        model_kwargs={
                            "cond_embed": audio,
                            "one_hot": one_hot,
                            "template": template,
                        }
                    )['loss']

                    loss = torch.mean(loss)
                    valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)

        val_losses.append(current_loss)
        os.makedirs(save_path, exist_ok=True)
        if e == args.max_epoch or e % 1 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'{args.model}_{args.dataset}_{e}.pth'))
            plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))
        print("epcoh: {}, current loss:{:.8f}".format(e + 1, current_loss))
        sys.stdout.flush()
    plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the end time
    print(f"End time: {end_time}")
    sys.stdout.flush()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
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
    parser.add_argument("--result_path", type=str, default="results", help='path to the predictions')
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
    parser.add_argument("--skip_steps", type=int, default=0, help='number of diffusion steps to skip during inference')
    parser.add_argument("--num_samples", type=int, default=1, help='number of samples to generate per audio')
    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the start time
    print(f"Start time: {start_time}")
    sys.stdout.flush()
    assert torch.cuda.is_available()
    diffusion = create_gaussian_diffusion(args)

    print("FaceDiffuser training on 3DMEAD")
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

    model = model.to(cuda)
    # resume training: check if a checkpoint exists and load it
    checkpoint_path = r"outputs/model/x_n.pth"
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Resuming from epoch n")
        sys.stdout.flush()

    dataset = get_dataloaders(args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    trainer_diff(args, dataset["train"], dataset["valid"], model, diffusion, optimizer,
                 epoch=args.max_epoch, device=args.device)


if __name__ == "__main__":
    main()