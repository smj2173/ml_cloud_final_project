from flask import Flask, render_template, request
from torchvision import models, transforms
from PIL import Image
import torch

app = Flask(__name__)




# TODO: Have to add .device(cuda) or something like that at the end of some statements for it to run on the gpu




# Load the PyTorch models
imagenet = torch.load('model1.pth')
damagenet = torch.load('model2.pth')

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
    # TODO: May have to preprocess the iamge?
    image = Image.open(image_file)
    imagenet_pred = model1(image)
    damagenet_pred = model2(image)
    
    # Print out the outputs
    result_html = f"<p>Model 1 Classification: {output1}</p>"
    result_html += f"<p>Model 2 Classification: {output2}</p>"
    
    # Render the new html
    return result_html

if __name__ == '__main__':
    app.run(debug=True)
