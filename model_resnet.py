import sys
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

def main():
    if len(sys.argv) != 2:
        print("Usage: python model_resnet.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    model.to(device)

    # preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    # load image
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    # predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    labels = weights.meta["categories"]
    top5 = torch.topk(probs, 5)

    print("\n=== RESNET PREDICTION ===")
    for idx, score in zip(top5.indices[0], top5.values[0]):
        print(f"{labels[idx]} : {score.item():.4f}")

if __name__ == "__main__":
    main()

