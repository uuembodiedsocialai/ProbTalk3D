import librosa
import argparse
import torch
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from models import FaceDiff
from transformers import Wav2Vec2Processor
import time
import pickle

from utils import *

emo_dict = {
       "0": "neutral",  # only have one intensity level
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


def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = FaceDiff(
        args,
        vertice_dim=args.vertice_dim,
        latent_dim=args.feature_dim,
        diffusion_steps=args.diff_steps,
        gru_latent_dim=args.gru_dim,
        num_layers=args.gru_layers
    )

    model_path = f'{args.save_path}/{args.model_name}_{args.dataset}_{args.epoch}.pth'
    print("Loading model from: ", model_path)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.data_path, args.template_path)

    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.id)
    one_hot = one_hot_labels[iter]
    id_one_hot = torch.FloatTensor(one_hot).to(device=args.device)
    emotion_one_hot = torch.eye(8)[args.emotion].to(device=args.device)
    intensity_one_hot = torch.eye(3)[args.intensity].to(device=args.device)
    one_hot = torch.cat([id_one_hot, emotion_one_hot, intensity_one_hot], dim=0)

    print("Loading templates for", args.subject)
    temp = templates[args.subject]

    template = temp.reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)   # [1, 15069]

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    start_time = time.time()
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
    audio_feature = processor(speech_array, return_tensors="pt", padding="longest",
                              sampling_rate=sampling_rate).input_values
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    diffusion = create_gaussian_diffusion(args)
    num_frames = int(audio_feature.shape[0] / sampling_rate * args.output_fps)
    num_frames -= 1
    # use ddim
    prediction=diffusion.ddim_sample_loop(
        model,
        (1, num_frames, args.vertice_dim),
        clip_denoised=False,
        model_kwargs={
            "cond_embed": audio_feature,
            "one_hot": one_hot,
            "template": template,
        },
        skip_timesteps=args.skip_steps,     # skip 900 timesteps
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        device=args.device,
    )
    prediction = prediction.squeeze()
    prediction = prediction.cpu().detach().numpy()

    elapsed = time.time() - start_time
    print("Inference time for ", prediction.shape[1], " frames is: ", elapsed, " seconds.")
    print("Inference time for 1 frame is: ", elapsed / prediction.shape[1], " seconds.")
    print("Inference time for 1 second of audio is: ", ((elapsed * args.fps) / prediction.shape[1]), " seconds.")

    emo = emo_dict[str(args.emotion)]  # retrieve emotion
    ints = int_dict[str(args.intensity)]  # retrieve intensity
    out_file_name = test_name + "_" + args.id + "_" + emo + "_" + ints
    print("Save file at: ", os.path.join(args.result_path, out_file_name))
    np.save(os.path.join(args.result_path, out_file_name), prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="face_diffuser")
    parser.add_argument("--epoch", type=str, default="50")
    parser.add_argument("--data_path", type=str, default=f"{parent_dir}/datasets/", help='name of the dataset folder. eg: BIWI')
    parser.add_argument("--dataset", type=str, default="mead", help='name of the dataset folder. eg: BIWI')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=256, help='GRU Vertex Decoder hidden size')
    parser.add_argument("--vertice_dim", type=int, default=15069, help='number of vertices - 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_subjects", type=str, default="M003 M005 M007 M009 M011 M012 M013 M019 "
                                                              "M022 M023 M024 M025 M026 M027 M028 M029 "
                                                              "M030 M031 W009 W011 W014 W015 W016 W018 "
                                                              "W019 W021 W023 W024 W025 W026 W028 W029")
    parser.add_argument("--test_subjects", type=str, default="M003 M005 M007 M009 M011 M012 M013 M019 "
                                                             "M022 M023 M024 M025 M026 M027 M028 M029 "
                                                             "M030 M031 W009 W011 W014 W015 W016 W018 "
                                                             "W019 W021 W023 W024 W025 W026 W028 W029")
    parser.add_argument("--wav_path", type=str, default=f"{parent_dir}/results/generation/test_audio/angry.wav",
                        help='path of the input audio signal in .wav format')
    parser.add_argument("--save_path", type=str, default="outputs/model", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="results/generation", help='path of the predictions in .npy format')
    parser.add_argument("--id", type=str, default="M009", help='select a conditioning subject from train_subjects')
    parser.add_argument("--emotion", type=int, default="6", help='select a conditioning subject from train_subjects')
    parser.add_argument("--intensity", type=int, default="1", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M009",
                        help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--template_path", type=str, default="templates_mead_vert.pkl",
                        help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates",
                        help='path of the mesh in BIWI topology')
    parser.add_argument("--input_fps", type=int, default=50,
                        help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=25,
                        help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--diff_steps", type=int, default=1000)
    parser.add_argument("--gru_dim", type=int, default=256)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--skip_steps", type=int, default=900)
    args = parser.parse_args()

    test_model(args)


if __name__ == "__main__":
    main()