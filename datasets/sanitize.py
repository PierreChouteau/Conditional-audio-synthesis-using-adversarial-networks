import os
import torchaudio

root_dir = "/fast-1/datasets/drums_ml/drums_ml_22"


def sanitize(root_dir):
    audio_dir = os.path.join(root_dir)
    filenames = os.listdir(root_dir)

    for i in range(len(filenames)):
        audio_path = os.path.join(audio_dir, filenames[i])

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.size(1) == 0:
            print(i, audio_path)
            os.remove(audio_path)

        else:
            print(i)
