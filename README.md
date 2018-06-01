# The Project:

This repository is code for a convolutional neural network that trains over the frames of a video  and log-spectrograms of the video's sound to generate subsequent frames.
There are three main models.
ENET - a model that takes two consequtive frames and generates the next frame.
YNET - a model that takes a frame and that frame's sound's log-spectrogram 


## Environment setup:

This code is supposed to be run in a Windows environment (since all of the file paths are according to the Windows format rather than Linux). The code will also benefit greatly in terms of speed from an installed NVidia GPU. This code was tested on a PC (running Windows 10) with an NVidia GeForce GTX 970 with CUDA version 8.0

There are two environments that you need to run this code, one for generating the log-spectrograms (called "spect") and the other for running the bulk of the code(called "tensorflow"). The key difference between the two environmets is the version of numpy. The following screenshot detail the two anaconda enviroments:

## Training the model:
1. to train enet, run the enet.py in a "tensorflow" environment. This will, by default, train the enet model over 2000 epochs and will save the weights after every 200 epochs.
2. to train ynet, run the ynet.py in a "tensorflow" environment. This will, by default, train the ynet model over 2000 epochs and will save the weights after every 200 epochs.
3. to train yenet, you first need to generate output of the previously two trained models, and save them under \data\tabletennis\fcnet\e and \data\tabletennis\fcnet\y for enet and ynet respectively. Then, run the fcnet.py in a "tensorflow" environment. This will, by default, train the fcnet model over 2000 epochs and will save the weights after every 100 epochs.

## Testing the model:
See the sample test file structure on what the test files are expected to be in each folder. Use the log_spec.py file in a "spect" environment to generate spectrograms from .wav files
To test the models, run emodel_test_feedback.py, ymodel_test_feedback.py and fmodel_test_feedback.py (in that order) to produce the final results in the testing\with_feeback\1\f\ dir. In this folder, 1 and 2 were provided as is, while 3 and 4 were generated from 1 and 2. 5 and 6 were used to generate 7 and 8, and so on.
