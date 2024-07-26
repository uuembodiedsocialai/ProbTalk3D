
# This folder will contain the data before preprocessing. 

### For folder "MEAD_audio_source":

1. Download the audios from the original MEAD dataset [here](https://wywu.github.io/projects/MEAD/MEAD.html) and put all the subjects' folders in "MEAD_audio_source" folder.

2. Extract the "audio.tar" archives for all the subjects in their respective folders.

3. Run the python script "process_audio.py"

4. The processed audio files will be saved in "~/datasets/mead/wav/" directory as wav files.


### For folder "EMICA-MEAD_flame2020":

1. Download the reconstructed 3DMEAD data from [here](https://download.is.tue.mpg.de/emote/mead_25fps/processed/reconstruction_v1.zip). Thanks to the authors of EMOTE for making it available. If the link is not working, please contact the EMOTE paper authors. 

2. Extract the "reconstruction_v1.zip" archive and put the "EMICA-MEAD_flame2020" folder in this directory.

3. Run the python script "process_animation.py"

4. The processed animation data files will be saved in "~/datasets/mead/param/" directory as npy files.



NOTE: If necessary, install the python libraries needed for the scripts to run. Check the imports in the scripts. 