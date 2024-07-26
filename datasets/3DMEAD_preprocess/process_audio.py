import numpy as np
import os, sys
from pydub import AudioSegment

# create list of subjects and sentences
subjects = os.listdir('MEAD_audio_source')
print("Total subjects: ", len(subjects))
seqs = os.listdir('MEAD_audio_source/M003/audio/neutral/level_1/')
print("Total sentences: ", len(seqs))

# emotion dict
emo_dict = {
    "0" : "neutral",
    "1" : "happy",
    "2" : "sad",
    "3" : "surprised",
    "4" : "fear",
    "5" : "disgusted",
    "6" : "angry",
    "7" : "contempt"
}

# intensity dict
intensity_dict = {
    "0" : "level_1",
    "1" : "level_2",
    "2" : "level_3",
}

# Read m4a data and create wavs for all subjects
folder = "MEAD_audio_source/"
output_folder = "../mead/wav/"
for subject in subjects:
    for emotion in range(len(emo_dict)):
        for intensity in range(len(intensity_dict)):
            for seq in seqs:
#                 print(folder + subject + "/front/" + emo_dict[str(emotion)] + "/" + intensity_dict[str(intensity)] + "/" + seq)
                main_file_path = folder + subject + "/audio/" + emo_dict[str(emotion)] + "/" + intensity_dict[str(intensity)] + "/" + seq
                try:
                    track = AudioSegment.from_file(main_file_path)
                except OSError:
                    # print(main_file_path, " was not found! But it is fine...")
                    continue
                output_filename = subject + "_" + seq.split('.')[0] + "_" + str(emotion) + "_" + str(intensity) + ".wav"
                output_filepath = output_folder + output_filename
                file_handle = track.export(output_filepath, format='wav')