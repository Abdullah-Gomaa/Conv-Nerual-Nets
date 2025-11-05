import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
import os

from model import CNN
from load_dataset import load_dataset

class_names = ['plane', 'car','bird','cat','deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


new_transform = transforms.Compose(
[
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]
)

def load_img(img_path):
    img = Image.open(img_path)
    img = new_transform(img)
    img = img.unsqueeze(0)
    return img


def load_model(model_path):
    loaded_cnn = CNN()
    loaded_cnn.load_state_dict(torch.load(model_path))
    return loaded_cnn


def predict(loaded_cnn, images_paths):
    images = [load_img(img) for img in images_paths]
    predictions = []
    with torch.no_grad():
        for img in images:
            outputs = loaded_cnn(img)
            _, pred = torch.max(outputs,1)
            
            predictions.append(pred.item())
            print(f"Prediction: {class_names[pred.item()]}")

    return predictions

def visualize_predictions(images_paths, predictions,indices=None):
    if indices is None:
        indices = range(len(images_paths))
    
    plt.figure(figsize=(12, 6))
    
    for i, idx in enumerate(indices):
        img = Image.open(images_paths[idx])
        label = class_names[predictions[idx]]

        plt.subplot(1, len(indices), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{label}", fontsize=20, color='red')

    plt.tight_layout()
    plt.savefig("predictions.png")
    
if __name__=="__main__":
    loaded_model = load_model("Models/latest_cnn.pth")
    images_paths  = ["airplane_ex1.jpeg","dog_ex2.jpeg","frog_ex3.jpeg"]
    predictions = predict(loaded_model,images_paths) 
    visualize_predictions(images_paths, predictions)