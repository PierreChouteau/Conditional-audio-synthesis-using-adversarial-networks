# Conditional-audio-synthesis-using-adversarial-networks


## Setup du projet
### Prérequis - Installation de miniconda (ou Anaconda)

On va utiliser minconda pour gérer notre environnement et les dépendances python, il faut donc l'installer. Pour cela, deux options: 
- Suivre les instructions d'installation du [site offiel](!https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)  
- Sinon, suivre les lignes de commandes ci-dessous: 

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

### Installation de l'environnement Conda

Maintenant que miniconda est bien installé, on peut créé l'environnement qui va servir à lancer nos entrainements de GANSynth:
```
conda create --name nomdeENV python=3.10.8
```

Puis pour activer son environnement: 
```
conda activate nomdeENV
```

Une fois, l'environnement activé, il doit y avoir (nomdeENV) au début de votre terminal ! Si tout est bon, on peut passer à l'installation des dépendances python: 
```
pip install -r requirement.txt
```

## Abstract

In recent years, Generative Adversarial Networks (GANs) have yielded impressive results on various
generative tasks. In particular, conditional GANs have gained increasing interest in user-guided generation,
yet tend to suffer from mode collapse, resulting in low sample diversity. In this project, you will study
how GANs can be used for class-conditional audio synthesis, and how to increase the diversity of the
generated sounds. Specifically, the ultimate goal is to obtain a model capable of generating audio based
on the desired genre. To do so, you will need to find a proper classification for "genres" and think about
how to accurately condition your model. Hence, this project tackles several challenges: the training of a
model on a variety of dataset and the conditioning of this model in order to produce a diversity of audio
within a genre.


## References 

[1] Jesse H. Engel et al. “GANSynth: Adversarial Neural Audio Synthesis”. In: CoRR abs/1902.08710 (2019). arXiv: 1902.08710. url: http://arxiv.org/abs/1902.08710.  
[2] Ian Goodfellow et al. “Generative adversarial nets”. In: Advances in neural information processing systems. 2014, pp. 2672–2680.  
[3] D. Griffin and Jae Lim. “Signal estimation from modified short-time Fourier transform”. In: IEEE Transactions on Acoustics, Speech, and Signal Processing 32.2 (1984), pp. 236–243. doi: 10.1109/TASSP.1984.1164317.  
[4] Diederik P Kingma and Max Welling. Auto-Encoding Variational Bayes. 2013. doi: 10.48550/ARXIV.1312.6114. url: https://arxiv.org/abs/1312.6114.  
[5] Qi Mao et al. “Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis”. In: CoRR abs/1903.05628 (2019). arXiv: 1903.05628. url: http://arxiv.org/abs/1903.05628.  
[6] Mehdi Mirza and Simon Osindero. “Conditional Generative Adversarial Nets”. In: CoRR abs/1411.1784 (2014). arXiv: 1411 . 1784. url: http://arxiv.org/abs/1411.1784.  
[7] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. 2015. doi: 10.48550/ARXIV.1511.06434. url: https://arxiv.org/abs/1511.06434.  