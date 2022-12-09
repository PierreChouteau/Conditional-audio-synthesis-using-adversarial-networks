import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from .helper import *


class NSynthDataset(Dataset):
    def __init__(self, root_dir, usage = 'train', maxi=1.25):
        self.root_dir = root_dir
        train_valid_test = {
            'train' : 'nsynth-train',
            'test' : 'nsynth-test',
            'valid' : 'nsynth-valid',
        }
        
        self.set_dir = os.path.join(self.root_dir, train_valid_test[usage])
        self.audio_dir = os.path.join(self.set_dir, 'audio')
        self.file_names = os.listdir(self.audio_dir)

        self.labels = json.load(open(os.path.join(self.set_dir,'examples.json')))
        self.maxi = maxi
       
       
    def __len__(self):
        return len(self.file_names)
    

    def resampling(self, waveform, sample_rate, resample_rate):
        
        resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype, lowpass_filter_width=128, rolloff=0.99, resampling_method="kaiser_window")
        resampled_waveform = resampler(waveform)
        
        return resampled_waveform


    def waveform_to_mel(self, waveform, n_fft, sample_rate, resample_rate, temps_sig):
        #-------------------------Ouverture d'un fichier et affichage spectro/waveform
        # waveform, sample_rate = torchaudio.load(SAMPLE_WAV_PATH)
        waveform = waveform

        #-----------------------Conversion de Sr
        resampled_waveform = self.resampling(waveform, sample_rate, resample_rate)
        
        #-----------------------Decoupe du signal resample a la bonne duree
        dur1 = len(resampled_waveform[0])
        durVoulue = int(temps_sig*resample_rate)
        if durVoulue <= dur1:
            resampled_waveform = resampled_waveform[:durVoulue]
        else:
            resampled_waveform = torch.cat((resampled_waveform, torch.zeros(1,durVoulue-dur1)), dim=1)

        #----------------------Spectro de Mel
        n_mels = 128
        hop_length = 63
        
        transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            center=True,
            pad_mode="reflect",
            hop_length=hop_length,
            pad=250,
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
            )
        melspec = transform(resampled_waveform)/ (self.maxi*0.8)
        return melspec
    
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.file_names[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        resample_rate = 16000
        n_fft = 1024
        temps_sig = 4
        
        melspec = self.waveform_to_mel(waveform, n_fft, sample_rate, resample_rate, temps_sig) / (self.maxi*0.8)
        label = self.labels[self.file_names[idx][:-4]]
        
        return melspec, label['instrument_family']
        
        
        
#-----------------------Reconstruction par Griffin-Lim
def mel_to_waveform(self, melspec, n_fft, resample_rate, maxi):
    melspec=melspec*(maxi*0.8)

    inverse_melscale_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1) #On repasse en STFT pour Griffin-Lim
    rev_spec = inverse_melscale_transform(melspec)
    
    # Declaration de Griffin-Lim
    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
    )
    waveform_r = griffin_lim(rev_spec) # On applique Griffin-Lim
    return waveform_r
        


        
# TEST = False
# if TEST:
#     # Get the list of all files and directories
#     path = "/home/pierre/OneDrive/ATIAM/UE - IM/Conditional-audio-synthesis-using-adversarial-networks/data/percussion"
#     #dir_list = os.listdir(path)
#     #print("Files and directories in '", path, "' :")
#     #print(dir_list)

#     resample_rate = 16000
#     n_fft = 1024
#     temps_sig = 2

#     dataset,maxi = dataloader(path,temps_sig,resample_rate,n_fft)

#     signal = mel_to_waveform(dataset[40],n_fft,resample_rate,maxi)
#     plot_spectrogram(dataset[0][0].numpy(), title='Reconstruct spectro', ylabel='freq_bin', aspect='auto', xmax=None)
#     plot_waveform(signal, resample_rate, title="Reconstruct", xlim=None, ylim=None)
#     play_audio(signal, resample_rate)

TEST = False
if TEST:
    
    loader = NSynthDataset("/fast-1/atiam22-23/nsynth", usage = 'train')
    
    maxi = 0 
    for i, (mel_spec, label) in enumerate(loader):
        abso = abs(mel_spec.numpy())
        max1 = np.amax(abso[0])
        maxi = max(maxi, max1)
        if i > 50000:
            print(maxi)
            break
        
        
    # maxi=0 
    # melspecs=[]
    # for i in range(0,len(dir_list)): #Obtention de maxi : à mettre dans le config ?
    #     melspec=loader.__getitem__(i)
    #     melspecs.append(melspec)

    #     abso=abs(melspec.numpy())
    #     max1=np.amax(abso[0])
    #     maxi=max(maxi,max1)