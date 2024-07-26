import h5py
import numpy as np
import os

# create list of subjects and sentences
subjects = os.listdir('EMICA-MEAD_flame2020')
print("Total subjects: ", len(subjects))
seqs = os.listdir('EMICA-MEAD_flame2020/M003/front/neutral/level_1/')
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

# Read data and create npys
appearance_file = 'appearance.hdf5'
main_file = "shape_pose_cam.hdf5"
folder = "EMICA-MEAD_flame2020/"
output_folder = "../mead/param/"
for subject in subjects:
    for emotion in range(len(emo_dict)):
        for intensity in range(len(intensity_dict)):
            for seq in seqs:
#                 print(folder + subject + "/front/" + emo_dict[str(emotion)] + "/" + intensity_dict[str(intensity)] + "/" + seq + "/" + main_file)
                main_file_path = folder + subject + "/front/" + emo_dict[str(emotion)] + "/" + intensity_dict[str(intensity)] + "/" + seq + "/" + main_file
                try:
                    f = h5py.File(main_file_path, "r")
                except OSError:
                    continue
#                 print(f["global_pose"][()].shape)
#                 print(subject + "_" + seq + "_" + str(emotion) + "_" + str(intensity) + ".npy")
                shape = f["shape"][()]
                exp = f["exp"][()]
                jaw = f["jaw"][()]
                out = np.concatenate((shape, exp), axis=2)
                out = np.concatenate((out, jaw), axis=2)
                # print(out.shape) (1, frames, params = 403 (first 300 are shape param, second 100 is exp params and last 3 are jaw params))
                output_filename = subject + "_" + seq + "_" + str(emotion) + "_" + str(intensity) + ".npy"
                output_filepath = output_folder + output_filename
                np.save(output_filepath, out)