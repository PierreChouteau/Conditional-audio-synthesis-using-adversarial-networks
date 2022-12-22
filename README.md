# Conditional audio synthesis using adversarial networks

In this project, we adapt the GANSynth model to work with a drums dataset. 
We also propose a variant to condition the training in order to obtain a model capable of generating audio based
on the desired genre.


## Install

To run our scripts, you will need to have a specific environment which will require the installation of miniconda (or anaconda). 
If you do not already have it, you can install it from the original [website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).


- Clone the github repository

``` 
git clone https://github.com/PierreChouteau/Conditional-audio-synthesis-using-adversarial-networks
``` 

- Create a virtual env with python 3.10.8:

``` 
conda create -n gansynth python=3.10.8
``` 

- Activate the environment:
``` 
conda activate gansynth
``` 

- Go into the repository and install the dependencies: 
``` 
cd Conditional-audio-synthesis-using-adversarial-networks
pip install -r requirements.txt
``` 

## Training

To start a default training (with the default configuration), simply run the script train.py: 

``` 
python train.py
``` 

If you want to modify the configuration (model_name, optimizers, etc...), you will have to change the config file (```default_config.yaml``` ), with the parameters you want. 


## Project Structure

```bash 
Conditional-audio-synthesis-using-adversarial-networks
├── configs    
│   └── __init__.py
│   └── config.py
├── data   
├── datasets   
│   └── __init__.py
│   └── dataset.py
│   └── helper.py
├── models
│   └── __init__.py
│   └── discriminator.py
│   └── gansynth.py
│   └── generator.py
├── notebooks
├── runs
├── trained_model
├── .gitignore
├── default_config.yaml
├── README.md
├── requirements.txt
└── train.py