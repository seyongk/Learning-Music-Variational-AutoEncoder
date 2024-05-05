# Music Variational AutoEncoder (MusicVAE) in PyTorch

This repository contains a PyTorch implementation of the Music Variational AutoEncoder (MusicVAE) model, as described in the paper ["A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music"](https://arxiv.org/abs/1803.05428) by Roberts et al.

MusicVAE is a deep learning model that learns a hierarchical representation of music and can generate new musical sequences. It combines a variational autoencoder (VAE) with a hierarchical decoder to capture long-term structure in music.

## Features

- Implements the MusicVAE model architecture in PyTorch
- Trains the model on MIDI data to learn a latent representation of music
- Generates new musical sequences by sampling from the learned latent space
- Provides utility functions for processing MIDI files and converting between MIDI and tensor representations

## Dataset

The MIDI data used in this repository is sourced from [arman-aminian/lofi-generator](https://github.com/arman-aminian/lofi-generator). It consists of a collection of MIDI files that can be used to train the MusicVAE model.

## Usage

1. Clone the repository: ```git clone https://github.com/yourusername/Learning-Music-Variational-AutoEncoder.git```

2. Install the required dependencies: ```pip install -r requirements.txt```

3. Prepare the MIDI data:
- Make a `midi_songs` folder and put your MIDI files into it.

4. Run the Jupyter notebook `main.ipynb` to train the MusicVAE model and generate new musical sequences.

## Code Structure

- `midi_utils.py`: Contains utility functions for processing MIDI files and converting between MIDI and tensor representations.
- `model.py`: Defines the MusicVAE model architecture using PyTorch.
- `loss.py`: Implements the loss function used for training the MusicVAE model.
- `main.ipynb`: Jupyter notebook that demonstrates loading MIDI data, training the MusicVAE model, and generating new musical sequences.

## Credits

- The MusicVAE model implementation is based on the paper ["A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music"](https://arxiv.org/abs/1803.05428) by Roberts et al.
- The MIDI utility functions and MIDI data are sourced from [arman-aminian/lofi-generator](https://github.com/arman-aminian/lofi-generator).

## Citation

If you use this code or the MusicVAE model in your research, please cite the following paper:  
@inproceedings{roberts2018hierarchical,
title={A hierarchical latent vector model for learning long-term structure in music},
author={Roberts, Adam and Engel, Jesse and Raffel, Colin and Hawthorne, Curtis and Eck, Douglas},
booktitle={International conference on machine learning},
pages={4364--4373},
year={2018},
organization={PMLR}
}

## License

This project is licensed under the [MIT License](LICENSE).
