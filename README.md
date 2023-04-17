# Image Coloring with Encoder-Decoder CNN based Deep Learning Model Streamlit-based Webapp
This project aims to develop an image coloring system that can color grayscale images using a deep learning encoder-decoder CNN model. The model is trained on the Kaggle Landscape Pictures dataset, which contains grayscale landscape images and their corresponding colored images. A web application is then created using Streamlit, which allows users to upload their own grayscale images and colorize them using the trained model.

### Requirements
This project requires the following libraries:

Python 3.x\
TensorFlow 2.x\
NumPy\
Matplotlib\
Streamlit\
pandas\
These libraries can be easily installed using pip or conda package manager.

### Dataset
The project uses the Kaggle Landscape Pictures dataset, which can be downloaded from the following link: https://www.kaggle.com/arnaud58/landscape-pictures.\

The dataset contains 10,000 grayscale images and their corresponding colored images. The grayscale images are of size 256x256 pixels, while the colored images are of size 512x512 pixels.

### Model Architecture
The model architecture used in this project is an encoder-decoder CNN model. The encoder is a convolutional neural network, which extracts features from the input grayscale image. The decoder is another convolutional neural network, which uses the extracted features to generate the output colored image. The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer.

#### Usage
To train the model, run the abovejupyter notebook file and save the model

#### streamlit web application
streamlit run app.py
The web application will be launched in the default web browser, and users can upload their own grayscale images and colorize them using the trained model.

### Acknowledgments
This project is based on the research paper "Colorful Image Colorization" by Zhang et al. (https://arxiv.org/pdf/1603.08511.pdf). The implementation of the encoder-decoder CNN model is inspired by the TensorFlow tutorial on Image Colorization (https://www.tensorflow.org/tutorials/generative/colorization).

### Conclusion
This project demonstrates the use of deep learning encoder-decoder CNN models for image coloring tasks. The model achieves decent colorization quality on the Kaggle Landscape Pictures dataset and can be further improved with more training data and fine-tuning of hyperparameters. The Streamlit-based web application provides an easy-to-use interface for users to colorize their own grayscale images.
