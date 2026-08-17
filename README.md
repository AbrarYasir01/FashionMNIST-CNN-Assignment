
# FashionMNIST CNN Image Classification

A complete **Convolutional Neural Network (CNN)** image-classification project built with **PyTorch**, using the **FashionMNIST** dataset and real-world custom images for inference.

The project demonstrates the complete deep-learning workflow — from dataset loading and preprocessing to CNN training, evaluation, visualization, model saving, and prediction on custom images.

---

## 📌 Project Overview

The objective of this project is to train a CNN to classify images into the **10 FashionMNIST clothing categories**, and then test the trained model on **10 custom clothing/accessory images**.

The project uses:

* **PyTorch** for deep learning
* **Torchvision** for FashionMNIST and image transformations
* **PIL** for custom-image loading
* **Matplotlib** for visualization
* **Seaborn** for the confusion matrix
* **Google Colab** for training and experimentation
* **GitHub** for project and model storage

---

## 🎯 Objectives

The main objectives are:

1. Load the FashionMNIST dataset automatically.
2. Preprocess the image data using `torchvision.transforms`.
3. Build a CNN using `torch.nn`.
4. Train the model using the Adam optimizer.
5. Use Cross Entropy Loss for classification.
6. Evaluate the model on validation and test data.
7. Generate training/validation loss and accuracy plots.
8. Generate a confusion matrix.
9. Save the trained model as a `.pth` state dictionary.
10. Test the model on real-world custom images.
11. Display predictions and confidence scores.
12. Perform visual error analysis.

---

## 🧠 Dataset

The project uses the **FashionMNIST** dataset.

FashionMNIST contains grayscale images of clothing and fashion-related objects.

Each image has a resolution of:

```text
28 × 28 pixels
```

There are **10 classes**:

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

### Dataset Split

The project uses:

```text
Training samples   : 48,000
Validation samples : 12,000
Testing samples    : 10,000
```

---

## 🏗️ CNN Architecture

The implemented CNN consists of two convolutional blocks followed by fully connected layers.

```text
Input
  │
  ▼
Conv2D
1 → 32 channels
3 × 3 kernel
  │
  ▼
ReLU
  │
  ▼
MaxPooling
  │
  ▼
Conv2D
32 → 64 channels
3 × 3 kernel
  │
  ▼
ReLU
  │
  ▼
MaxPooling
  │
  ▼
Flatten
  │
  ▼
Linear
3136 → 128
  │
  ▼
ReLU
  │
  ▼
Linear
128 → 10
  │
  ▼
Class Prediction
```

### Model Definition

```python
CNN(
  (features): Sequential(
    Conv2d(1, 32, kernel_size=3, padding=1)
    ReLU()
    MaxPool2d(2)

    Conv2d(32, 64, kernel_size=3, padding=1)
    ReLU()
    MaxPool2d(2)
  )

  (classifier): Sequential(
    Flatten()
    Linear(3136, 128)
    ReLU()
    Linear(128, 10)
  )
)
```

### Model Parameters

```text
Total parameters     : 421,642
Trainable parameters : 421,642
```

---

## 🔄 Image Preprocessing

FashionMNIST images are grayscale and have a size of 28×28 pixels.

The preprocessing pipeline is:

```text
Image
  ↓
Resize to 28 × 28
  ↓
Convert to Tensor
  ↓
Normalize
  ↓
CNN
```

The normalization uses:

```python
MEAN = (0.5,)
STD = (0.5,)
```

The normalization formula is:

[
x' = \frac{x-\mu}{\sigma}
]

For custom images, an additional grayscale conversion is performed:

```text
Custom RGB Image
       ↓
Grayscale
       ↓
Resize 28 × 28
       ↓
ToTensor
       ↓
Normalize
       ↓
CNN
```

This ensures that custom images have the same basic input format as FashionMNIST.

---

## ⚙️ Training Configuration

The model was trained for:

```text
Epochs      : 10
Optimizer   : Adam
Learning Rate : 0.001
Loss        : CrossEntropyLoss
```

The training process follows the standard PyTorch workflow:

```python
optimizer.zero_grad()

output = model(images)

loss = criterion(output, labels)

loss.backward()

optimizer.step()
```

---

## 📊 Training Results

The final training results were:

| Metric              |     Result |
| ------------------- | ---------: |
| Training Accuracy   | **96.85%** |
| Validation Accuracy | **91.72%** |
| Test Accuracy       | **90.89%** |

### Final Loss

```text
Training Loss   : 0.0874
Validation Loss : 0.2644
```

---

## 📈 Training History

The model was trained for 10 epochs.

| Epoch | Train Loss | Train Acc. | Val Loss | Val Acc. |
| ----: | ---------: | ---------: | -------: | -------: |
|     1 |     0.4675 |     83.16% |   0.3393 |   87.90% |
|     2 |     0.2927 |     89.42% |   0.2807 |   89.97% |
|     3 |     0.2439 |     91.15% |   0.2475 |   90.84% |
|     4 |     0.2100 |     92.30% |   0.2421 |   91.38% |
|     5 |     0.1859 |     93.24% |   0.2423 |   90.97% |
|     6 |     0.1606 |     94.10% |   0.2254 |   91.88% |
|     7 |     0.1415 |     94.80% |   0.2318 |   91.93% |
|     8 |     0.1206 |     95.56% |   0.2571 |   91.38% |
|     9 |     0.1023 |     96.22% |   0.2503 |   91.51% |
|    10 |     0.0874 |     96.85% |   0.2644 |   91.72% |

The training accuracy consistently increased throughout training, while validation accuracy stabilized around 91–92%.

---

## 📊 Confusion Matrix

The model achieved a **90.89% test accuracy** on the FashionMNIST test set.

The confusion matrix is used to analyze how predictions are distributed among the ten classes.

The model performed particularly well on classes such as:

* Trouser
* Sandal
* Sneaker
* Bag

Some confusion can be observed between visually similar clothing categories, particularly:

* T-shirt/top ↔ Shirt
* Pullover ↔ Coat
* Shirt ↔ T-shirt/top
* Ankle boot ↔ Sneaker/Sandal

---

## 📱 Custom Image Testing

To evaluate real-world performance, **10 custom images** were added to the repository.

The images are:

```text
ankleboot.jpg
bag.jpg
coat.jpg
dress.jpg
pullover.jpg
sandal.jpg
shirt.jpg
sneaker.jpg
trouser.jpg
tshirt.jpg
```

These images were processed using the same basic preprocessing approach before being passed to the trained CNN.

---

## 🔍 Custom Image Predictions

The model produced the following predictions:

| Image         | Prediction  | Confidence |
| ------------- | ----------- | ---------: |
| ankleboot.jpg | Bag         |     65.72% |
| bag.jpg       | Bag         |     65.33% |
| coat.jpg      | Bag         |     99.98% |
| dress.jpg     | Bag         |    100.00% |
| pullover.jpg  | Bag         |    100.00% |
| sandal.jpg    | Trouser     |     98.36% |
| shirt.jpg     | Bag         |     63.71% |
| sneaker.jpg   | Bag         |    100.00% |
| trouser.jpg   | Trouser     |     97.03% |
| tshirt.jpg    | T-shirt/top |     72.70% |

The custom-image results demonstrate an important aspect of real-world image classification: **high performance on the standard dataset does not necessarily guarantee high performance on photographs from a different visual domain**.

FashionMNIST consists of small grayscale images, whereas the custom photographs contain real objects, different backgrounds, lighting conditions, textures, orientations, and image compositions.

---

## 🖼️ Custom Prediction Gallery

The notebook generates a prediction gallery showing:

* Custom image
* Predicted class
* Confidence percentage

Example:

```text
Pred: Bag
Confidence: 65.72%
```

This provides a visual representation of the model's real-world predictions.

---

## ❌ Error Analysis

The project also performs error analysis using incorrectly classified FashionMNIST test images.

Examples include:

```text
True: Sneaker
Pred: Sandal
```

```text
True: Coat
Pred: Pullover
```

```text
True: Ankle boot
Pred: Sandal
```

These errors are useful for understanding which classes are difficult for the CNN to distinguish.

---

## 💾 Saved Model

The trained model is saved as a PyTorch state dictionary:

```text
model/220140.pth
```

The model can be restored using:

```python
model.load_state_dict(
    torch.load("model/220140.pth")
)
```

The model file contains the learned parameters rather than the entire training notebook.

---

## 📁 Repository Structure

The GitHub repository follows this structure:

```text
FashionMNIST-CNN-Assignment/
│
├── dataset/
│   ├── ankleboot.jpg
│   ├── bag.jpg
│   ├── coat.jpg
│   ├── dress.jpg
│   ├── pullover.jpg
│   ├── sandal.jpg
│   ├── shirt.jpg
│   ├── sneaker.jpg
│   ├── trouser.jpg
│   └── tshirt.jpg
│
├── model/
│   └── 220140.pth
│
├── 220140.ipynb
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AbrarYasir01/FashionMNIST-CNN-Assignment.git
```

### 2. Open the notebook

Open:

```text
220140.ipynb
```

in Google Colab or Jupyter Notebook.

### 3. Run the notebook

The notebook automatically:

1. Clones/updates the GitHub repository.
2. Downloads FashionMNIST using `torchvision`.
3. Creates the training and validation datasets.
4. Applies preprocessing.
5. Builds the CNN.
6. Trains the model.
7. Evaluates the model.
8. Generates the confusion matrix.
9. Loads the custom images.
10. Predicts their classes.
11. Displays confidence scores.
12. Performs error analysis.

No manual upload of the custom images is required during notebook execution.

---

## 🧪 Technologies Used

| Technology   | Purpose                             |
| ------------ | ----------------------------------- |
| Python       | Programming language                |
| PyTorch      | CNN implementation and training     |
| Torchvision  | Dataset and image transformations   |
| NumPy        | Numerical operations                |
| Matplotlib   | Visualization                       |
| Seaborn      | Confusion matrix visualization      |
| PIL          | Image loading and processing        |
| Google Colab | Development/training environment    |
| GitHub       | Version control and project storage |

---

## 📌 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Machine Learning
* Deep Learning
* Image Classification
* Convolutional Neural Networks
* Convolution
* ReLU activation
* Max Pooling
* Flattening
* Fully Connected Layers
* Cross Entropy Loss
* Adam Optimization
* Backpropagation
* Gradient Descent
* Training/Validation Split
* Image Preprocessing
* Normalization
* Model Evaluation
* Confusion Matrix
* Softmax Probabilities
* Confidence Scores
* Error Analysis
* Model Serialization using `.pth`

---

## ⚠️ Limitations

The model achieves **90.89% accuracy on the FashionMNIST test set**, but its performance on custom photographs is substantially weaker.

This is expected because the two datasets differ significantly.

FashionMNIST:

```text
28 × 28
Grayscale
Centered objects
Simple backgrounds
```

Custom photographs:

```text
Higher resolution
Real-world lighting
Different backgrounds
Different object orientations
Real textures and colors
```

Converting a photograph to 28×28 grayscale also removes considerable visual information.

Therefore, the custom-image experiment demonstrates the **domain gap** between a benchmark dataset and real-world images.

---

## 🔮 Possible Improvements

Future versions could improve real-world performance by:

* Adding data augmentation.
* Training with more real-world clothing photographs.
* Using transfer learning.
* Using a deeper CNN architecture.
* Increasing input resolution.
* Using a dataset containing real-world fashion photographs.
* Applying better background preprocessing.
* Using dataset-specific normalization statistics.
* Fine-tuning the model using custom images.

---

## 👨‍💻 Author

**Abrar Yasir**

GitHub:

**AbrarYasir01**

Repository:

**FashionMNIST-CNN-Assignment**

---

## 📄 Academic Project

This project was developed as part of a **CNN Image Classification with PyTorch** assignment demonstrating the complete workflow from standard dataset training to real-world image testing.

---

### ⭐ Results Summary

```text
============================================================
FASHIONMNIST CNN CLASSIFICATION
============================================================

Training Accuracy   : 96.85%
Validation Accuracy : 91.72%
Test Accuracy       : 90.89%

Training Epochs     : 10
Custom Images       : 10

Loss Function       : CrossEntropyLoss
Optimizer           : Adam
Learning Rate       : 0.001

Total Parameters    : 421,642
Trainable Parameters: 421,642

Model               : model/220140.pth

============================================================
```
