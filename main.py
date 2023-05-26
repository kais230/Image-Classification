import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Image path and threshold for object and animal detection
image_path = "object3.jpg"
confidence_threshold = 0.5

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with Image.open(image_path) as image:
    input_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(input_image)

# Load the labels for ImageNet classes
with open("imagenet_labels.txt") as label_file:
    labels = label_file.read().splitlines()

# Extract predicted class and confidence score
predicted_class_index = torch.argmax(predictions)
predicted_class = labels[predicted_class_index]
confidence_score = torch.softmax(predictions, dim=1)[0, predicted_class_index].item()

# Create an output image
output_image = image.copy()
draw = ImageDraw.Draw(output_image)
font = ImageFont.truetype("arial.ttf", size=50)

if confidence_score > confidence_threshold:
    # Add the predicted label and confidence percentage
    label = f"Class: {predicted_class} ({confidence_score * 100:.2f}%)"
    draw.text((25, 25), label, fill='green', font=font)
else:
    # Add message for low confidence in animal or object prediction
    draw.text((10, 10), "Could not identify the animal or object with confidence", fill='red', font=font)

# Save and display the output image
output_image.save("output_image.jpg")
output_image.show()
