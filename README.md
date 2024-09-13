## **ProbTalk3D**
Official PyTorch implementation for the paper:

> **ProbTalk3D: Non-Deterministic Emotion Controllable Speech-Driven 3D Facial Animation Synthesis Using VQ-VAE. (Accepted at [ACM SIGGRAPH MIG 2024](https://sgmig.hosting.acm.org/mig-2024/))**
>
> <a href='https://uuembodiedsocialai.github.io/ProbTalk3D/'><img src='https://img.shields.io/badge/Project-Website-blue'></a> <a href='https://arxiv.org/pdf/2409.07966'><img src='https://img.shields.io/badge/Paper-red'></a> <a href='https://arxiv.org/pdf/2409.07966'><img src='https://img.shields.io/badge/arXiv-[]-red'></a> <a href='https://uuembodiedsocialai.github.io/ProbTalk3D/#video-container'><img src='https://img.shields.io/badge/Project-Video-Green'></a> 
> 
> We propose ProbTalk3D, a VQ-VAE based probabilistic model for emotion controllable speech-driven 3D facial animation synthesis. ProbTalk3D first learns a motion prior using VQ-VAE codebook matching, then trains a speech and emotion conditioned network leveraging this prior. During inference, probabilistic sampling of latent codebook embeddings enables non-deterministic outputs.

## **Environment**
<details><summary>Click to expand</summary>

### System Requirement
- Linux and Windows (tested on Windows 10)
- Python 3.9+
- PyTorch 2.1.1
- CUDA 12.1 (GPU with at least 2.55GB VRAM)

### Virtual Environment
To run our program, first create a virtual environment. We recommend using [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) or [miniforge](https://conda-forge.org/miniforge/). Once Miniconda or Miniforge is installed, open Command Prompt (make sure to run it as Administrator on Windows) and run the following commands:
```
conda create --name probtalk3d python=3.9
conda activate probtalk3d
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, navigate to the project `root` folder and execute:

```
pip install -r requirements.txt
```
</details>

## **Dataset**
<details><summary>Click to expand</summary>

Download 3DMEAD dataset following the instruction of [EMOTE](https://github.com/radekd91/inferno/tree/release/EMOTE/inferno_apps/TalkingHead/data_processing). This dataset represents facial animations using FLAME parameters.

### Data Download and Preprocess 
- Please refer to the `README.md` file in `datasets/3DMEAD_preprocess/` folder. 
- After processing, the resulting `*.npy` files will be located in the `datasets/mead/param` folder, and the `.wav` files should be in the `datasets/mead/wav` folder.

- <b> Optional Operation </b>
    <details><summary>Click to expand</summary>
    
    For training the comparison model in vertex space, we provide a script to transfer the FLAME parameters to vertices. Execute the script `pre_process/param_to_vert.py`. The resulting `*.npy` files should be located in the `datasets/mead/vertex` folder.
    </details>
</details>


## **Model Training**
<details><summary>Click to expand</summary>
To train the model from scratch, follow the 2-stage training approach outlined below.

### Stage 1
For the first stage of training, use the following commands:
- On Windows and Linux:
    ```
    python train_all.py experiment=vqvae_prior state=new data=mead_prior model=model_vqvae_prior
    ```
- If the Linux system has Slurm Workload Manager, use the following command: 
    ```
    sbatch train_vqvae_prior.sh
    ```

- <b> Optional Operation </b>
    <details><summary>Click to expand</summary>

    - We use Hydra configuration, which allows us to easily override settings at runtime. For example, to change the GPU ID to 1 on a multi-GPU system, set `trainer.devices=[1]`. To load a small amount of data for debugging, set `data.debug=true`.
    - To resume training from a checkpoint, set the `state` to resume and specify the `folder` and `version`. Specifically, replace the `folder` and `version` in the command below with the folder name where the checkpoint is saved. Our program generates a random name for each run, and the version is assigned automatically by the program, which may vary depending on the operating system.
        ```
        python train_all.py experiment=vqvae_prior state=resume data=mead_prior model=model_vqvae_prior folder=outputs/MEAD/vqvae_prior/XXX version=0
        ```
- <b> VAE variant </b>
    <details><summary>Click to expand</summary>  
  
    To train the VAE variant for comparison, follow the same instructions as above and change the `model` setting as below:
    ```
    python train_all.py experiment=vae_prior state=new data=mead_prior model=model_vae_prior
    ```
    </details>

### Stage 2
After completing stage 1 training, execute the following command to proceed with stage 2 training. Set `model.folder` and `model.version` to the location where the motion prior checkpoint is stored:
- On Windows and Linux:
    ```
    python train_all.py experiment=vqvae_pred state=new data=mead_pred model=model_vqvae_pred model.folder_prior=outputs/MEAD/vqvae_prior/XXX model.version_prior=0
    ```
- If the Linux system has Slurm Workload Manager, use the following command. Remember to revise the `model.folder_prior` and `model.version_prior` in the file. 
    ```
    sbatch train_vqvae_pred.sh
    ```
- <b> Optional Operation </b>
    <details><summary>Click to expand</summary>
  
    - Similar to the first stage of training, the GPU ID can be changed by setting `trainer.devices=[1]`, and debug mode can be enabled by setting `data.debug=true`.
    - To resume training from a checkpoint, set the state to `resume` and specify the `folder` and `version`: 
        ```
        python train_all.py experiment=vqvae_pred state=resume data=mead_pred model=model_vqvae_pred folder=outputs/MEAD/vqvae_pred/XXX version=0 model.folder_prior=outputs/MEAD/vqvae_prior/XXX model.version_prior=0
        ```
    </details>
- <b> VAE variant </b>
    <details><summary>Click to expand</summary>
  
    To train the VAE variant for comparison, follow the same instructions as above and change the `model` setting as below:
    ```
    python train_all.py experiment=vae_pred state=new data=mead_pred model=model_vae_pred model.folder_prior=outputs/MEAD/vae_prior/XXX model.version_prior=0
    ```
    </details>
</details>


## **Evaluation**
<details><summary>Click to expand</summary>

Download the trained model weights from [HERE](https://drive.google.com/file/d/1U29vNIh0Ig74YjqBNBr57saKll9H5LVX/view?usp=sharing) and unzip them into the project `root` folder.

### Quantitative Evaluation
We provide code to compute the evaluation metrics mentioned in our paper. To evaluate our trained model, run the following:
- On Windows and Linux:
    ```
    python evaluation.py folder=model_weights/ProbTalk3D/stage_2 number_of_samples=10
    ```
- If the Linux system has Slurm Workload Manager, use the following command:
    ```
    sbatch evaluation.sh
    ```
- <b> Optional Operation </b>
  <details><summary>Click to expand</summary>
  
  - Adjust the GPU ID if necessary; for instance, set `device=1`.
  - To evaluate your own trained model, specify the `folder` and `version` according to the location where the checkpoint is saved:
    ```
    python evaluation.py folder=outputs/MEAD/vqvae_pred/XXX version=0 number_of_samples=10
    ```
  </details>

- <b> VAE variant </b>
    <details><summary>Click to expand</summary>
  
    To evaluate the trained VAE variant, execute the following command:
    ```
    python evaluation.py folder=model_weights/VAE_variant/stage_2 number_of_samples=10
    ```
    </details>

### Qualitative Evaluation
For qualitative evaluation, refer to the script `evaluation_quality.py`.

</details>


## **Animation Generation**
<details><summary>Click to expand</summary>

Download the trained model weights from [HERE](https://drive.google.com/file/d/1U29vNIh0Ig74YjqBNBr57saKll9H5LVX/view?usp=sharing) and unzip them into the project `root` folder.

### Generate Prediction
Our model is trained to generate animations across 32 speaking styles (IDs), 8 emotions, and 3 intensities. Check all available conditions:

<details><summary>Click to expand</summary>
ID: 

```
M003, M005, M007, M009, M011, M012, M013, M019,
M022, M023, M024, M025, M026, M027, M028, M029,
M030, M031, W009, W011, W014, W015, W016, W018,
W019, W021, W023, W024, W025, W026, W028, W029
```
emotion:

```
neutral, happy, sad, surprised, fear, disgusted, angry, contempt
```
intensity (stands for low, medium, high intensity in order):

```
0, 1, 2
```
</details>

We provide several test audios. Run the following command to generate animations (with a random style) using the trained ProbTalk3D. This will produce `.npy` files that can be rendered into videos.
- On Windows and Linux:
  ```
  python generation.py folder=model_weights/ProbTalk3D/stage_2 input_path=results/generation/test_audio
  ```
- To specify styles for the provided test audios, use the following command. When setting style conditions for multiple files at once, ensure the setting order follows the filename sorting of Windows.
  ```
  python generation.py folder=model_weights/ProbTalk3D/stage_2 input_path=results/generation/test_audio id=[\"M009\",\"M024\",\"W023\",\"W011\",\"W019\",\"M013\",\"M011\",\"W016\"]  emotion=[\"angry\",\"contempt\",\"disgusted\",\"fear\",\"happy\",\"neutral\",\"sad\",\"surprised\"] intensity=[1,2,0,2,1,0,1,2] 
  ```
- <b> Optional Operation </b>
  <details><summary>Click to expand</summary>

  - To generate multiple outputs (for example, 2 outputs) using one test audio, set `number_of_samples=2`.
  - The default generation process uses stochastic sampling. To control diversity, adjust `temperature=X`. The default X value is 0.2; we recommend choosing between 0.1 to 0.5.
  - Our model can operate deterministically by setting `sample=false`, bypassing stochastic sampling.
  - To play with your own data, modify the `input_path` or place them in the folder `results/generation/test_audio`.
  - Adjust the GPU ID if necessary; for instance, set `device=1`.
  - To generate animation with your own trained model, specify the `folder` and `version` according to the location where the checkpoint is saved:
    ```
    python generation.py folder=outputs/MEAD/vqvae_pred/XXX version=0 input_path=results/generation/test_audio
    ```
  </details>

- <b>  VAE variant </b>
  <details><summary>Click to expand</summary>
  
  - To generate animations (with a random style) using the trained VAE variant, run the following command:
  
    ```
    python generation.py folder=model_weights/VAE_variant/stage_2 input_path=results/generation/test_audio
    ```
  - Similarly, follow the above instructions to specify the style or generate multiple files by setting `number_of_samples`.
  - The default generation process sets the scale factor to 20. To control diversity, adjust `fact=X`. We recommend setting X between 1 and 40. Setting X as 1 means no scaling.
   </details>
    
### Render
The generated `.npy` files contain FLAME parameters and can be rendered into videos following the below instructions. 
- We use blender to render the predicted motion. First, download the dependencies from [HERE](https://drive.google.com/file/d/1EJ0enL27YbybzUAQ3olFGhkNpEfiaoU2/view?usp=sharing) and extract them into the `deps` folder. Please note that this command can only be executed on Windows:
  ```
  python render_param.py result_folder=results/generation/vqvae_pred/stage_2/0.2 audio_folder=results/generation/test_audio
  ```
- <b> Optional Operation </b>
  <details><summary>Click to expand</summary>
  
  - To play with your own data, modify `result_folder` to where the generated `.npy` files are stored, and `audio_folder` to where the `.wav` files are located.
  - We provide post-processing code in the `post_process` folder. To change face shapes for the predicted motion, refer to the script `change_shape_param.py`.
  - To convert predicted motion to vertex space, refer to the script `post_process/transfer_to_vert.py`. For rendering animation in vertex space, use the following command on Windows and Linux: 
    ```
    python render_vert.py result_folder=results/generation/vqvae_pred/stage_2/0.2 audio_folder=results/generation/test_audio
    ```
  </details>

- <b> VAE variant </b>
  <details><summary>Click to expand</summary>

  To render the generated animations produced by the trained VAE variant, use the following command on Windows:
  ```
  python render_param.py result_folder=results/generation/vae_pred/stage_2/20 audio_folder=results/generation/test_audio
  ```
  </details>
</details>


## **Comparison**
<details><summary>Click to expand</summary>

For comparing with the diffusion model FaceDiffuser (modified version), navigate to the `diffusion` folder.
### Model training
To train the model from scratch, execute the following command:
```
python main.py
```
### Evaluation
To quantitatively evaluate our trained FaceDiffuser model, run the following command:
```
python evaluation_facediff.py --save_path "../model_weights/FaceDiffuser" --max_epoch 50
```
### Animation Generation

#### Generate Prediction
To generate animations using our trained model, execute the following command. Modify the path and style settings as needed.
```
python predict.py --save_path "../model_weights/FaceDiffuser" --epoch 50 --subject "M009" --id "M009" --emotion 6 --intensity 1 --wav_path "../results/generation/test_audio/angry.wav"
```
#### Render
Navigate back to the project `root` folder and run the following command:
```
python render_vert.py result_folder=diffusion/results/generation audio_folder=results/generation/test_audio
```
</details>

</details>

## Citation ## 
If you find the code useful for your work, please consider starring this repository and citing it:
```
@misc{wu2024probtalk3dnondeterministicemotioncontrollable,
      title={ProbTalk3D: Non-Deterministic Emotion Controllable Speech-Driven 3D Facial Animation Synthesis Using VQ-VAE}, 
      author={Sichun Wu and Kazi Injamamul Haque and Zerrin Yumak},
      year={2024},
      eprint={2409.07966},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.07966}, 
}
        
% PS: The inproceedings bibtex will be available upon getting the DOI from ACM. 

```

## **Acknowledgements**

We borrow and adapt the code from [Learning to Listen](https://github.com/evonneng/learning2listen), [CodeTalker](https://github.com/Doubiiu/CodeTalker), [TEMOS](https://github.com/Mathux/TEMOS),   [FaceXHuBERT](https://github.com/galib360/FaceXHuBERT), [FaceDiffuser](https://github.com/uuembodiedsocialai/FaceDiffuser). We appreciate the authors for making their code available and facilitating future research. Additionally, we are grateful to the creators of the 3DMEAD datasets used in this project.

Any third-party packages are owned by their respective authors and must be used under their respective licenses.

## **License**
This repository is released under [CC-BY-NC-4.0-International License](https://github.com/Gibberlings3/GitHub-Templates/blob/master/License-Templates/CC-BY-NC-4.0/LICENSE-CC-BY-NC-4.0.md).
