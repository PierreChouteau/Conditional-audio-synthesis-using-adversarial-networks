#ToDo : 4s, 16kHz, spectro, Mel (torchaudio),normaliser avec max de la banque, revenir Ã  l'audio avec Griffin-Lim
#Normalisation entre -0.8 et 0.8
#Dénormaliser en reconstruct et resample en fonction
#gbittencourt pour le dataset
#Faire comparaison en plot le spectro avant/après Griffin
#Faire un .py helper plot
#Faire tourner dataload pour avoir maxi puis update maxi
#Source tutos : https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

import torchaudio.functional as F
import torchaudio.transforms as T

import math
import timeit
import librosa
import resampy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from IPython.display import Audio, display
from torchaudio.utils import download_asset
# import OS module
import os

from helper_plot import * #Import de mes fonctions

#plt.close('all')

  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#------------------------------------Mes Fonctions

#-------------------------------------------------------------NSynth
class NSynthDataset(Dataset):
    def __init__(self, root_dir, usage = 'train', transform = None):
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
        self.transform = transform
       
    def __len__(self):
        return len(self.file_names)
    
    #-----------------------DataLoader complet
    def dataloader(path,temps_sig,resample_rate,n_fft):
        dir_list = os.listdir(path)
        maxi=0
        melspecs=[]
        for i in range(1,len(dir_list)):
            SAMPLE_WAV_PATH=dir_list[i]
            
            #Uniquement si max sur la waveform
            #maxi=getmax(path)
            #print(i)
            melspec=transfo(path+"/"+SAMPLE_WAV_PATH,n_fft,resample_rate,temps_sig,maxi=1.25)
            melspecs.append(melspec) #i-1 sur mon PC, pas sur le dataload
            
            #Uniquement si max sur le Mel Spectro
            abso=abs(melspec.numpy())
            max1=np.amax(abso[0])
            maxi=max(maxi,max1)
        #melspecs2=np.empty(i,dtype=object)
        for u in melspecs:
            u=u.clone().detach().requires_grad_(True)/(maxi*0.8)
        
        return melspecs,maxi
    
    def resampling(waveform, sample_rate, resample_rate):
        resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype,lowpass_filter_width=128,rolloff=0.99, resampling_method="kaiser_window")
        resampled_waveform = resampler(waveform)
        return resampled_waveform

    def transfo(waveform,sample_rate,n_fft,resample_rate,temps_sig,maxi=1.25):
        #-------------------------Ouverture d'un fichier et affichage spectro/waveform
        
        waveform=waveform/(maxi*0.8)
        #print_stats(waveform, sample_rate=sample_rate)
        #plot_waveform(waveform, sample_rate)
        #plot_specgram(waveform, sample_rate)
        #play_audio(waveform, sample_rate)
        
        #-----------------------Conversion de Sr
        
        resampled_waveform=resampling(waveform,sample_rate,resample_rate)
        
        #-----------------------Découpe du signal resamplé à la bonne durée
        dur1=len(resampled_waveform[0])
        #print(resample_rate)
        durVoulue=int(temps_sig*resample_rate)
        #print(dur1,durVoulue)
        #print(resampled_waveform.shape)
        if durVoulue<=dur1:
            resampled_waveform=resampled_waveform[:durVoulue]
            #print(1)
            #print(resampled_waveform.shape)
        else:
            #print(2)
            resampled_waveform=torch.cat((resampled_waveform,torch.zeros(1,durVoulue-dur1)),dim=1)
            #print(resampled_waveform.shape)
        #plot_specgram(resampled_waveform, resample_rate)
        #plot_waveform(resampled_waveform, resample_rate)
        
        
        
        #----------------------Spectro de Mel
        
        #hop_length = 512
        #win_length = 990
        n_mels = 128
        
        transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            #win_length=win_length,
            #hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
            )
        melspec = transform(resampled_waveform)  # (channel, n_mels, time)
        #plot_spectrogram(
        #    melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')
        return melspec

    def getmax(path):
        dir_list = os.listdir(path)
        maxi=0
        for i in range(1,len(dir_list)):
            SAMPLE_WAV_PATH=dir_list[i]
            waveform, sample_rate = torchaudio.load(path+"/"+SAMPLE_WAV_PATH)
            abso=abs(waveform).numpy()
            max1=np.amax(abso[0])
            maxi=max(maxi,max1)
        return maxi
        
        
    def __getitem__(self, idx, maxi=1.25):
        audio_path = os.path.join(self.audio_dir, self.file_names[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        resample_rate = 16000
        n_fft = 1024
        temps_sig=2
        
        melspec=transfo(waveform,sample_rate ,n_fft,resample_rate,temps_sig,maxi=1.25)/(maxi*0.8)
        #Il faut réfléchir à comment intégrer maxi ? Faire tourner tout le dataloader une fois d'abord en entier pour récupérer le max puis le mettre en input ?
        
        label = self.labels[self.file_names[idx][:-4]]
        
        #print(label)
        return melspec, label['instrument_family']

#-----------------------Reconstruction par Griffin-Lim
def recons(melspec,n_fft,resample_rate,maxi):
    
    melspec=melspec*(maxi*0.8)

    inverse_melscale_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1) #On repasse en STFT pour Griffin-Lim
    rev_spec = inverse_melscale_transform(melspec)
    
    griffin_lim = T.GriffinLim(#Déclaration de Griffin-Lim
        n_fft=n_fft,
        #win_length=win_length,
        #hop_length=hop_length,
    )
    waveform_r = griffin_lim(rev_spec)#On applique Griffin-Lim
    
    plot_waveform(waveform_r, resample_rate, title="Reconstructed")
    return waveform_r


#-----------------------------------------------------------Code

#-----------------------DataLoader complet





# Get the list of all files and directories
path = "C:/Users/GaHoo/Desktop/Cours/ATIAM/2. Informatique/Projet/data/percussion"
#dir_list = os.listdir(path)
#print("Files and directories in '", path, "' :")
#print(dir_list)
loader=NSynthDataset(path)

maxi=0
melspecs=[]
for i in range(1,len(dir_list)):
    #Uniquement si max sur la waveform
    #maxi=getmax(path)
    #print(i)
    melspec=loader.__getitem__(i)
    melspecs.append(melspec)
    
    #Uniquement si max sur le Mel Spectro
    abso=abs(melspec.numpy())
    max1=np.amax(abso[0])
    maxi=max(maxi,max1)
    


#signal=recons(dataset[40],n_fft,resample_rate,maxi)
    
    





