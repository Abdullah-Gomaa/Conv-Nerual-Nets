import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import os

from model import CNN
from load_dataset import load_dataset

def train(learning_rate=0.001, epochs = 15):
    train_loader,_ = load_dataset() 

    cnn_model = CNN()
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

    n_steps = len(train_loader)
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (imgs,labels) in enumerate(train_loader):
            
            # Forward prop
            outputs = cnn_model(imgs)

            # Calc loss
            loss = loss_func(outputs,labels)

            # Backward prop

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"[{epoch+1}] loss: {running_loss/n_steps:.3f}")

    print("Finished Training !")

    save_latest_model(cnn_model,'./Models')


def save_latest_model(model, save_dir='.'):
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define base model filename
    base_name = 'latest_cnn.pth'
    base_path = os.path.join(save_dir, base_name)

    # If a latest model already exists, rename it
    if os.path.exists(base_path):
        name, ext = os.path.splitext(base_name)
        i = 1
        while True:
            new_name = f"{name}_{i}{ext}"
            new_path = os.path.join(save_dir, new_name)
            if not os.path.exists(new_path):
                os.rename(base_path, new_path)
                print(f"Existing model renamed to: {new_name}")
                break
            i += 1

    # Save the latest model
    torch.save(model.state_dict(), base_path)
    print(f"Model saved as: {base_path}")


if __name__=="__main__":
    train()