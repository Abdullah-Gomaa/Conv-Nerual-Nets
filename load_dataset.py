import torch 
import torchvision
import torchvision.transforms as transforms


def load_dataset(batchsize=64):
    transfrom = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(root='./dataset',train=True,download=True,transform=transfrom)
    test_dataset = torchvision.datasets.CIFAR10(root='./dataset',train=False,download=True,transform=transfrom)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batchsize, shuffle=False)

    return train_loader,test_loader


if __name__ == "__main__":
    train_loader,test_loader = load_dataset()
   
    # --- Dataset overview ---
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    print("\n Dataset Information:")
    print(f"  • Training samples: {len(train_dataset)}")
    print(f"  • Test samples: {len(test_dataset)}")
    print(f"  • Classes: {train_dataset.classes}")

    # --- Inspect one sample ---
    img, label = train_dataset[0]
    print("\n Sample inspection:")
    print(f"  • Image tensor shape: {img.shape} (C, H, W)")
    print(f"  • Label index: {label}  ->  '{train_dataset.classes[label]}'")
    print(f"  • Image dtype: {img.dtype}")
    print(f"  • Image min/max after normalization: {img.min():.3f}, {img.max():.3f}")

    # --- Inspect one batch ---
    images, labels = next(iter(train_loader))
    print("\n Batch Information:")
    print(f"  • Batch size: {images.shape[0]}")
    print(f"  • Image batch shape: {images.shape} (B, C, H, W)")
    print(f"  • Label batch shape: {labels.shape}")
    print(f"  • First 10 labels in batch: {labels[:10].tolist()}")