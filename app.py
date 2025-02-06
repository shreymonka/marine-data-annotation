from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = Flask(__name__)
CORS(app)

def get_model(num_classes):
    model = models.efficientnet_v2_l(pretrained=True)
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers for fine-tuning
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, num_classes)  # 1280 is the output features for efficientnet_v2_l
    )
    return model

def load_model():
    classes = ['Fish', 'Goldfish', 'Harbor seal', 'Jellyfish', 'Lobster', 'Oyster', 'Sea turtle', 'Squid', 'Starfish']
    model = get_model(len(classes))
    model.load_state_dict(torch.load('marine_animal_classifier.pth', 
                                   map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model, classes

# Image preprocessing transform
# EfficientNetV2_L typically performs better with larger input sizes
transform = transforms.Compose([
    transforms.Resize((480, 480)),  # Increased size for better performance
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model and classes globally
model, classes = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read and preprocess the image
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img_copy = img.copy()
        
        # Transform and predict
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Use mixed precision inference
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction = classes[predicted.item()]
                confidence = confidence.item()
        
        # Draw prediction on image
        draw = ImageDraw.Draw(img_copy)
        text = f"{prediction}: {confidence:.2%}"
        draw.text((10, 10), text, fill='white', stroke_width=2, stroke_fill='black')
        
        # Convert image to base64
        buffered = io.BytesIO()
        img_copy.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'class': prediction,
            'confidence': float(confidence),
            'annotated_image': img_str
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)