from flask import Flask, render_template, request
import torchvision
from torchvision.models import ResNet18_Weights 
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn

app = Flask(__name__)

def get_model(model_path):
    # Load the model class
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Load the model weights and state from the input path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def transform_image(image):
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return transformer(image)

# Setup cuda device and show whether CPU or GPU is being used
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('Running tests on GPU')
else:
    print('Running tests on CPU')

# Load the PyTorch models and set them to eval
imagenet = get_model('outputs/imagenette_model.pth')
combined = get_model('outputs/combined_model.pth')
imagenet.eval()
combined.eval()

# Page without image uploaded
@app.route('/')
def index():
    return render_template('index.html')

# Page with image uploaded
@app.route('/classify', methods=['POST'])
def classify():
    # Error check if file exists
    if 'image' not in request.files:
        return "No image found in request!"

    # Error check if image was selected
    image_file = request.files['image']
    if image_file.filename == '':
        return "No image selected!"

    # Open image and get model outputs (classifications)
    image = Image.open(image_file)
    input_tensor = transform_image(image)
    input_tensor = input_tensor.to(device)
    input_batch = input_tensor.unsqueeze(0)
    _, imagenet_preds = torch.max(imagenet(input_batch).data, 1)
    _, combined_preds = torch.max(combined(input_batch).data, 1)
    
    # Classification dictionary
    classification = {0: 'Cathedral', 1: 'Dog', 2: 'Fish', 3: 'French Horn', 4: 'Garbage Truck', 5: 'Gas Pump', 6: 'Golf Ball', 7: 'Parachute', 8: 'Sawing', 9: 'Stereo'}

    # Print out the outputs
    result_html = f"<p>Imagenette Model Classification: <b>{classification[imagenet_preds.item()]}</b></p>"
    result_html += f"<p>Combined Imagenette and Damagenet Model Classification: <b>{classification[combined_preds.item()]}</b></p>"
    
    # Render the new html
    return result_html

if __name__ == '__main__':
    app.run(debug=True)
