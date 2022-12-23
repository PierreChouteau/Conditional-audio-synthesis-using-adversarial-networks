import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np
import os
import json
from .helper import *


class NSynthDataset(Dataset):
    def __init__(
        self,
        root_dir,
        usage="train",
        filter_key=None,
        resample_rate=16000,
        signal_duration=4,
        maxi=1,
    ):
        self.root_dir = root_dir
        train_valid_test = {
            "train": "nsynth-train",
            "test": "nsynth-test",
            "valid": "nsynth-valid",
        }

        self.set_dir = os.path.join(self.root_dir, train_valid_test[usage])
        self.audio_dir = os.path.join(self.set_dir, "audio")
        self.filenames = os.listdir(self.audio_dir)

        if filter_key != None:
            self.filenames = list(filter(lambda x: filter_key in x, self.filenames))

        self.labels = json.load(open(os.path.join(self.set_dir, "examples.json")))
        self.maxi = maxi

        self.resample_rate = resample_rate
        self.signal_duration = signal_duration

    def __len__(self):
        return len(self.filenames)

    def resampling(self, waveform, sample_rate, resample_rate):

        resampler = T.Resample(
            sample_rate,
            resample_rate,
            dtype=waveform.dtype,
            lowpass_filter_width=128,
            rolloff=0.99,
            resampling_method="kaiser_window",
        )
        resampled_waveform = resampler(waveform)

        return resampled_waveform

    def __waveform_to_mel__(self, waveform, sample_rate, resample_rate, temps_sig):
        # -------------------------Ouverture d'un fichier et affichage spectro/waveform
        waveform = waveform

        # -----------------------Conversion de Sr
        resampled_waveform = self.resampling(waveform, sample_rate, resample_rate)

        # -----------------------Decoupe du signal resample a la bonne duree
        dur1 = len(resampled_waveform[0])
        durVoulue = int(temps_sig * resample_rate)
        if durVoulue <= dur1:
            resampled_waveform = resampled_waveform[:durVoulue]
        else:
            resampled_waveform = torch.cat(
                (resampled_waveform, torch.zeros(1, durVoulue - dur1)), dim=1
            )

        # ----------------------Spectro de Mel
        # relation provenant du code officiel de GANSynth
        time_steps, n_mels = 128, 1024
        frame_length = n_mels * 2
        hop_length = int((1.0 - 0.75) * frame_length)
        num_samples = hop_length * (time_steps - 1) + frame_length

        transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=frame_length,
            hop_length=hop_length,
            win_length=frame_length,
            center=False,
            power=2.0,
            n_mels=n_mels,
        )
        resampled_waveform = torch.nn.functional.pad(
            resampled_waveform,
            ((num_samples - len(resampled_waveform[0]), 0)),
            "constant",
            0,
        )

        melspec = transform(resampled_waveform)  # (channel, n_mels, time)

        # Passage au log, avec la log1p
        melspec_log = torch.log(1 + melspec)
        return melspec_log

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.filenames[idx])
        waveform, sample_rate = torchaudio.load(audio_path)

        melspec_log = self.__waveform_to_mel__(
            waveform, sample_rate, self.resample_rate, self.signal_duration
        )
        melspec_log_norm = (melspec_log / (self.maxi / (2 * 0.8))) - 0.8

        label = self.labels[self.filenames[idx][:-4]]

        return melspec_log_norm, label["pitch"], melspec_log


class CustomDataset(Dataset):
    # Todo: Add label to this dataset
    def __init__(
        self,
        root_dir,
        resample_rate=16000,
        signal_duration=4,
        maxi=1,
    ):
        self.root_dir = root_dir

        self.audio_dir = os.path.join(self.root_dir)
        self.filenames = os.listdir(self.root_dir)

        self.maxi = maxi

        self.resample_rate = resample_rate
        self.signal_duration = signal_duration

    def __len__(self):
        return len(self.filenames)

    def __resampling__(self, waveform, sample_rate, resample_rate):
        # Todo: Make a separate script that does a complete resampling of the dataset
        resampler = T.Resample(
            sample_rate,
            self.resample_rate,
            dtype=waveform.dtype,
            lowpass_filter_width=128,
            rolloff=0.99,
            resampling_method="kaiser_window",
        )
        resampled_waveform = resampler(waveform)

        return resampled_waveform

    def __waveform_to_mel__(self, waveform, sample_rate, resample_rate, temps_sig):
        # -------------------------Ouverture d'un fichier et affichage spectro/waveform
        waveform = waveform

        # -----------------------Conversion de Sr
        # Todo: Make a separate script that does a complete resampling of the dataset.
        resampled_waveform = self.__resampling__(waveform, sample_rate, resample_rate)

        # -----------------------Decoupe du signal resample a la bonne duree
        # Todo: Make a separate script that allows to get a unique size of audio for the whole dataset.
        dur1 = len(resampled_waveform[0])
        durVoulue = int(temps_sig * resample_rate)
        if durVoulue <= dur1:
            resampled_waveform = resampled_waveform[:durVoulue]
        else:
            resampled_waveform = torch.cat(
                (resampled_waveform, torch.zeros(1, durVoulue - dur1)), dim=1
            )

        # ----------------------Spectro de Mel
        # relation from the official GANSynth code
        time_steps, n_mels = 128, 1024
        frame_length = n_mels * 2
        hop_length = int((1.0 - 0.75) * frame_length)
        num_samples = hop_length * (time_steps - 1) + frame_length

        transform = T.MelSpectrogram(
            sample_rate=resample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            win_length=frame_length,
            center=False,
            power=2.0,
            n_mels=n_mels,
        )
        resampled_waveform = torch.nn.functional.pad(
            resampled_waveform,
            ((num_samples - len(resampled_waveform[0]), 0)),
            "constant",
            0,
        )

        melspec = transform(resampled_waveform)  # (channel, n_mels, time)

        # Passage au log, avec la log1p
        melspec_log = torch.log(1 + melspec)
        return melspec_log

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.filenames[idx])

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            label = "drums"  # Todo: get the labels and incorporate them into the code

            melspec_log = self.__waveform_to_mel__(
                waveform, sample_rate, self.resample_rate, self.signal_duration
            )
            melspec_log_norm = (melspec_log / (self.maxi / (2 * 0.8))) - 0.8
            return melspec_log_norm, label, melspec_log

        except:
            pass


# -----------------------Reconstruction par Griffin-Lim
def mel_to_waveform(melspec_log_norm, maxi, device=torch.device("cpu")):

    time_steps, n_mels = 128, 1024
    frame_length = n_mels * 2
    hop_length = int((1.0 - 0.75) * frame_length)

    melspec_log = (melspec_log_norm + 0.8) * (maxi / (2 * 0.8))
    melspec = torch.exp(melspec_log) - 1

    # On repasse en STFT pour Griffin-Lim
    inverse_melscale_transform = T.InverseMelScale(
        n_stft=frame_length // 2 + 1, n_mels=n_mels, sample_rate=16000
    )
    inverse_melscale_transform.to(device)
    rev_spec = inverse_melscale_transform(melspec)

    # Declaration de Griffin-Lim
    griffin_lim = T.GriffinLim(
        n_fft=frame_length,
        n_iter=32,
        hop_length=hop_length,
        win_length=frame_length,
    )
    griffin_lim.to(device)
    waveform_r = griffin_lim(rev_spec)  # On applique Griffin-Lim

    # Normalisation de l'audio de sortie
    waveform_r = waveform_r / torch.max(torch.abs(waveform_r))
    return waveform_r


#######################################################################################
# Just a test function to verify the right behavior of the dataset
#######################################################################################
def test_data(TEST=False):
    if TEST:
        loader = NSynthDataset(
            "/fast-1/atiam22-23/nsynth", filter_key="acoustic", usage="train"
        )

        maxi = 0
        for i, (melspec_log_norm, label, melspec_log) in enumerate(loader):
            abso = np.abs(melspec_log.numpy())
            max1 = np.amax(abso[0])
            maxi = max(maxi, max1)
            if i > 500:
                print(maxi)
                break


# test_data(False)
