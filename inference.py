import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
import argparse


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def predict(image_path, model_path, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = len(class_names)
    
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return class_names[predicted_class.item()], confidence.item() * 100


def main():
    parser = argparse.ArgumentParser(description='Run inference on an image')
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='List of class names')
    
    args = parser.parse_args()
    
    predicted_class, confidence = predict(args.image, args.model, args.classes)
    
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == '__main__':
    main()