#ToDo : 4s, 16kHz, spectro, Mel (torchaudio),normaliser avec max de la banque, revenir Ã  l'audio avec Griffin-Lim
#Normalisation entre -0.8 et 0.8
#Source tutos : https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
import torch
import torchaudio
import numpy as np

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

plt.close('all')

#-----------------------------Fonctions Helper - Uniquement affichage
#Pour le modèle train, on ne devrait avoir besoin que de l'affichage des formes d'onde au fil des epoch

def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

#-----Afficher les formes d'onde en torchaudio
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

  #-----Pour les spectros en STFT
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)
  
  #-----Pour les spectro de Mel
def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)

#-----Play audio....
def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

#-----Plot Mel filter bank
def plot_mel_fbank(fbank, title=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Filter bank')
  axs.imshow(fbank, aspect='auto')
  axs.set_ylabel('frequency bin')
  axs.set_xlabel('mel bin')
  plt.show(block=False)
  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#------------------------------------Mes Fonctions

def transfo(SAMPLE_WAV_PATH,n_fft,resample_rate,temps_sig,maxi=1.25):
    #-------------------------Ouverture d'un fichier et affichage spectro/waveform
    
    #metadata = torchaudio.info(SAMPLE_WAV_PATH)
    #print(metadata)
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_PATH)
    waveform=waveform/(maxi*0.8)
    #print_stats(waveform, sample_rate=sample_rate)
    #plot_waveform(waveform, sample_rate)
    #plot_specgram(waveform, sample_rate)
    #play_audio(waveform, sample_rate)
    
    #-----------------------Conversion de Sr
    
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype,lowpass_filter_width=128,rolloff=0.99, resampling_method="kaiser_window")
    resampled_waveform = resampler(waveform)
    
    #-----------------------Découpe du signal resamplé à la bonne durée
    dur1=len(resampled_waveform[0])
    print(resample_rate)
    durVoulue=int(temps_sig*resample_rate)
    print(dur1,durVoulue)
    #print(resampled_waveform.shape)
    if durVoulue<=dur1:
        resampled_waveform=resampled_waveform[:durVoulue]
        print(1)
        print(resampled_waveform.shape)
    else:
        print(2)
        resampled_waveform=torch.cat((resampled_waveform,torch.zeros(1,durVoulue-dur1)),dim=1)
        print(resampled_waveform.shape)
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




#-----------------------Reconstruction par Griffin-Lim
def recons(melspec,n_fft,resample_rate):

    inverse_melscale_transform = T.InverseMelScale(n_stft=n_fft // 2 + 1) #On repasse en STFT pour Griffin-Lim
    rev_spec = inverse_melscale_transform(melspec)
    
    griffin_lim = T.GriffinLim(#Déclaration de Griffin-Lim
        n_fft=n_fft,
        #win_length=win_length,
        #hop_length=hop_length,
    )
    waveform_r = griffin_lim(rev_spec)#On applique Griffin-Lim
    
    plot_waveform(waveform_r, resample_rate, title="Reconstructed")
    
    
    
    
    
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
        
    
    
    
    return melspecs
    
#-----------------------------------------------------------Code

# Get the list of all files and directories
path = "C:/Users/GaHoo/Desktop/Cours/ATIAM/2. Informatique/Projet/data/percussion"
#dir_list = os.listdir(path)
#print("Files and directories in '", path, "' :")
# prints all files
#print(dir_list)


#SAMPLE_WAV_PATH = "data/percussion/agogo1.wav"
resample_rate = 16000
n_fft = 1024
temps_sig=2

dataset=dataloader(path,temps_sig,resample_rate,n_fft)

#melspec=transfo(SAMPLE_WAV_PATH,n_fft,resample_rate,maxi)


signal=recons(dataset[1],n_fft,resample_rate)
    
    





