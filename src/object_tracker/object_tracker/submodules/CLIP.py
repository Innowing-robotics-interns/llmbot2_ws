import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x64", device=device)

image = preprocess(Image.open("/home/fyp/Pictures/River.JPG")).unsqueeze(0).to(device)
text = clip.tokenize(["a river", "a dog"]).to(device)

print("enter loop")
for i in range(10):
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        print(image_features.shape)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


