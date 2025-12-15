import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import os


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def load_data(data_path, batch_size=32, train_split=0.8):
    transform = get_transforms()
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(dataset.classes)


def create_model(num_classes, use_gradient_checkpointing=True):
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model


def train_epoch(model, train_loader, optimizer, loss_fn, scaler, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images).logits
            loss = loss_fn(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images).logits
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, num_classes = load_data(
        args.data_path,
        args.batch_size,
        args.train_split
    )
    
    model = create_model(num_classes, args.gradient_checkpointing)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = CrossEntropyLoss()
    scaler = GradScaler()
    
    best_accuracy = 0.0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        val_accuracy = validate(model, val_loader, device)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_accuracy,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  New best model saved!")
        
        print()
    
    print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train Vision Transformer')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset folder')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/val split ratio')
    parser.add_argument('--step_size', type=int, default=2,
                        help='Learning rate scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate decay factor')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)


if __name__ == '__main__':
    main()