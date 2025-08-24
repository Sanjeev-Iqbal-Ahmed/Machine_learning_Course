# MNIST Digit Classification using TensorFlow

A simple neural network implementation for classifying handwritten digits (0-9) using the MNIST dataset and TensorFlow/Keras.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Code Explanation](#code-explanation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Key Concepts](#key-concepts)

## 🔍 Overview

This project implements a basic feedforward neural network to recognize handwritten digits from the MNIST dataset. The model achieves high accuracy in classifying digits 0-9 using a simple architecture with just two layers.

**What the code does:**
1. Loads and preprocesses the MNIST dataset
2. Builds a neural network with TensorFlow/Keras
3. Trains the model on 60,000 training images
4. Evaluates performance on 10,000 test images
5. Makes predictions and visualizes results

## 📊 Dataset

**MNIST (Modified National Institute of Standards and Technology)**
- **Training set**: 60,000 grayscale images (28×28 pixels)
- **Test set**: 10,000 grayscale images (28×28 pixels)
- **Classes**: 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
- **Pixel values**: Originally 0-255, normalized to 0-1 for better training

## 🏗️ Model Architecture

```
Input Layer (784 neurons)
    ↓
Dense Layer (128 neurons, ReLU activation)
    ↓
Output Layer (10 neurons, Softmax activation)
```

**Layer Details:**
- **Flatten Layer**: Converts 28×28 image matrix into 784-element vector
- **Hidden Layer**: 128 neurons with ReLU activation function
- **Output Layer**: 10 neurons (one per digit) with Softmax activation

## 💻 Code Explanation

### 1. Data Loading and Preprocessing
```python
mnist = tf.keras.datasets.mnist
(training_data, training_labels), (test_data, test_labels) = mnist.load_data()
training_data, test_data = training_data / 255, test_data / 255
```
- Loads the built-in MNIST dataset from TensorFlow
- Normalizes pixel values from 0-255 range to 0-1 range
- **Why normalize?** Helps the model train faster and more effectively

### 2. Model Construction
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```
- **Sequential**: Linear stack of layers
- **Flatten**: Reshapes 2D image data into 1D vector
- **Dense**: Fully connected layers
- **ReLU**: Rectified Linear Unit (f(x) = max(0, x))
- **Softmax**: Converts outputs to probability distribution

### 3. Model Compilation
```python
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
- **Adam Optimizer**: Adaptive learning rate optimization algorithm
- **Sparse Categorical Crossentropy**: Loss function for multi-class classification
- **Accuracy**: Metric to track during training

### 4. Training
```python
model.fit(training_data, training_labels, epochs=5)
```
- Trains the model for 5 epochs (complete passes through training data)
- Model learns to map input images to correct digit labels

### 5. Evaluation and Prediction
```python
model.evaluate(test_data, test_labels)
predictions = model.predict(test_data)
```
- Tests model performance on unseen data
- Generates probability predictions for each test image

### 6. Visualization
```python
plt.imshow(test_data[index], cmap='gray')
plt.title(f"Actual: {test_labels[index]}, Predicted: {np.argmax(predictions[index])}")
```
- Displays test image with actual vs predicted labels
- `np.argmax()` finds the class with highest probability

## 🛠️ Requirements

```
tensorflow>=2.0.0
numpy
matplotlib
```

Install with:
```bash
pip install tensorflow numpy matplotlib
```

## 🚀 Usage

1. **Run the script**: Execute the Python file to train and test the model
2. **Monitor training**: Watch accuracy improve over 5 epochs
3. **View results**: See test accuracy and sample prediction visualization

```bash
python mnist_classifier.py
```

## 📈 Results

**Expected Performance:**
- Training accuracy: ~98-99%
- Test accuracy: ~97-98%
- Training time: ~1-2 minutes on modern hardware

**Sample Output:**
```
Epoch 5/5
1875/1875 - 4s - loss: 0.0729 - accuracy: 0.9781
313/313 - 1s - loss: 0.0758 - accuracy: 0.9761
```

## 🧠 Key Concepts

### Neural Network Fundamentals
- **Neuron**: Basic processing unit that applies weighted sum + activation function
- **Layer**: Collection of neurons that process inputs in parallel
- **Forward Pass**: Data flows from input to output through layers
- **Backpropagation**: Algorithm that updates weights based on prediction errors

### Activation Functions
- **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`
  - Solves vanishing gradient problem
  - Computationally efficient
  - Most commonly used in hidden layers

- **Softmax**: `f(x_i) = e^(x_i) / Σ(e^(x_j))`
  - Converts raw scores to probabilities
  - Output sums to 1.0
  - Perfect for multi-class classification

### Loss Function
- **Sparse Categorical Crossentropy**: Used when labels are integers (not one-hot encoded)
- Measures difference between predicted and actual probability distributions
- Lower loss = better model performance

### Optimization
- **Adam Optimizer**: Combines benefits of AdaGrad and RMSProp
- Adaptive learning rates for each parameter
- Generally works well out-of-the-box

## 🔧 Customization Ideas

1. **Add more layers**: Increase model complexity
2. **Experiment with neurons**: Try different hidden layer sizes
3. **Add dropout**: Prevent overfitting with `tf.keras.layers.Dropout(0.2)`
4. **Change epochs**: Train for more/fewer iterations
5. **Try different optimizers**: SGD, RMSProp, etc.

## 📚 Learning Outcomes

After running this code, you'll understand:
- Basic neural network architecture
- Data preprocessing for machine learning
- Model training and evaluation workflow
- How to make predictions with trained models
- Visualization of ML results

---

**Note**: This is a beginner-friendly implementation focusing on core concepts rather than achieving state-of-the-art performance. For production use, consider adding regularization, validation sets, and more sophisticated architectures.