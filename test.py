import torch 

from model import CNN
from load_dataset import load_dataset


def test(model_path):
    _,test_loader = load_dataset()
    loaded_cnn = CNN()
    loaded_cnn.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)

        for imgs, labels in test_loader:

            outputs = loaded_cnn(imgs)

            _, preds = torch.max(outputs,1)
            
            n_correct += (preds == labels).sum().item()

    acc =(n_correct / n_samples) * 100.0
    print(f"Model Accuracy = {acc} %")


if __name__=="__main__":
    test("Models/latest_cnn.pth")