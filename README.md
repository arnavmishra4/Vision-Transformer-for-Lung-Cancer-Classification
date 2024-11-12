# Lung Cancer Classification using Vision Transformer (ViT)

## Project Overview

This project aims to classify lung cancer images into different categories using a **Vision Transformer (ViT)** model. The task involves training a deep learning model to classify lung cancer images into three categories:

- **Adenocarcinoma** (a type of lung cancer)
- **Benign** (non-cancerous tissue)
- **Squamous Cell Carcinoma** (another type of lung cancer)

We utilize a Vision Transformer (ViT), a transformer-based architecture designed for image classification, which has proven to be effective in handling image data by treating images as sequences of patches.

## Dataset

The dataset used for this project is the **Lung Cancer Histopathological Images** dataset, available on Kaggle. The dataset consists of histopathological images of lung cancer categorized into three classes:

1. **Adenocarcinoma**
2. **Benign**
3. **Squamous Cell Carcinoma**

The images are provided in three separate folders, each corresponding to one of the cancer types. The dataset can be accessed [here](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images).

## Project Structure

The project is divided into the following steps:

1. **Data Preprocessing**
    - The images are preprocessed to make them suitable for model training. This includes resizing images, normalizing pixel values, and converting labels into numerical values (0, 1, 2 for the three classes).
    - A split is made between training and validation data.

2. **Model Architecture**
    - The **Vision Transformer (ViT)** model is used as the core deep learning model for classification. This model splits images into patches, processes them via a transformer architecture, and then outputs predictions for each image.
    - The model was trained with **Cross-Entropy Loss** for multi-class classification.

3. **Training**
    - The model was trained using the **Adam optimizer** with a learning rate of 3e-5.
    - A **learning rate scheduler** (`StepLR`) is used to adjust the learning rate after every 2 epochs to improve convergence during training.

4. **Evaluation**
    - After training, the model is evaluated on the validation set, and the accuracy is reported. 

5. **Testing and Results**
    - The final model's accuracy on the validation set was printed after each epoch to track the model's performance and ensure that it generalizes well to unseen data.

## Dependencies

- `torch` (PyTorch)
- `torchvision`
- `tqdm`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `transformers` (Hugging Face, for Vision Transformer)

## How to Run the Project

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/arnavmishra4/Vision-Transformer-for-Lung-Cancer-Classification.git
cd Vision-Transformer-for-Lung-Cancer-Classification
