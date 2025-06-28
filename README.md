# MNIST Convolutional Neural Network (CNN) Classifier

This repository contains a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify handwritten digit images from the MNIST dataset.

## 🤖 What is a Convolutional Neural Network (CNN)?

A **Convolutional Neural Network (CNN)** is a type of deep learning model particularly effective for image classification and computer vision tasks. Unlike Feed Forward Neural Networks, CNNs:
- Automatically extract features using convolutional layers.
- Use pooling layers to reduce spatial dimensions and computation.
- Handle spatial hierarchies, making them highly effective for image recognition.

CNNs are widely used for:
- Handwritten digit recognition (MNIST)
- Object detection
- Image classification
- Face recognition

---

## 🚀 Features
✅ Loads the MNIST dataset (28x28 grayscale images of handwritten digits 0–9).  
✅ Normalizes pixel values for faster and stable training.  
✅ Uses a Convolutional Neural Network with:
- Multiple convolutional layers with ReLU activation.
- Max pooling layers to reduce image size.
- Dense layers with softmax output for classification.

✅ Trains the model and evaluates accuracy on the test set.  
✅ Plots training and validation accuracy/loss for performance analysis.  
✅ Displays test images with actual and predicted labels for visual verification.

---

## 🗂️ Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) is a classic benchmark dataset in machine learning and computer vision, containing:
- 60,000 training images
- 10,000 testing images
- 28x28 grayscale images of handwritten digits (0–9).

---

## 🛠️ Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

Install dependencies using:

```bash
pip install tensorflow keras matplotlib numpy
