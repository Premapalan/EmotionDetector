# Emotion Detector  ( Computer vision, DeepLearning with TensorFlow Fun project :) )

## Description
The aim of this project is to identify and detect different user facial expression in real time. I have used data from Kaggle to to train the model. You can find the data in the following link
```text 
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
```
I have built Convolutional Neral Network(CNN) on Tensorflow framework(open source deep learning framework). The model has been trained with 25 epochs(The number of epochs is a hyperparameter that determines the number of times the learning algorithm operates over the entire dataset of training) and built with 52% overall accuracy. The Emotion Detector was trained to detect the following emotions: 'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise‘. 


## Installation

### On Ubuntu

1. Set your python 3 as default if not.

    ```text
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
    sudo update-alternatives --config python
    ```

2. git clone https://gitlab.com/pprasathpp/emotion-detector_tensorflow.git
3. pip install virtualenv
4. enter command : virtualenv your_choice_env
5. source yout_choice_env/bin/activate
6. Check that you have a clean virtual environment

    ```text
    pip freeze
    ```

    you should get a empty line  if not check that your PYTHONPATH empty is, if not enter in your terminal PYTHONPATH="" and put the same command at the end of your .bashrc

7. pip install -r Requirements.txt
8. DONE  :blush:

## Usage

python3 emotion_detector.py

## samples
![Neutral](/images/Neutral.png)
![Happy](/images/Happy.png)
![Sad](/images/Sad.png)
![Angry](/images/Angry.png)
![Surprise](/images/Surprise.png)

## Current work

1. Detection of 7 different facial emotions ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise') based on training of 28709 images.
2. Model accuracy approximately 51%
