import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = 2

#credit: https://github.com/yunjey/pytorch-tutorial/
# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),      #in_channels, out_channels ,
            nn.ReLU(),     
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),      
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),    
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten()
        self.denseLayer1 = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),     
            nn.Dropout(0.1))
        self.denseLayer2 = nn.Sequential(
            nn.Linear(100, 30),
            nn.ReLU(),     
            nn.Dropout(0.1))
        self.denseLayer3 = nn.Sequential(
            nn.Linear(30, 10), 
            nn.ReLU(),     
            nn.Dropout(0.1))
        self.fc = nn.Linear(10, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        out = self.denseLayer1(out)
        out = self.denseLayer2(out)
        out = self.denseLayer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out
        
'''
INSTANTIATE MODEL CLASS
'''
model = ConvNet(num_classes).to(device)

# To enable GPU
model.to(device)

model.load_state_dict(torch.load('Project.pkl'))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def model_predict(img_path, model):
    img_file = Image.open(img_path).resize((64, 64))
    img_grey = img_file.convert('L')
    # img = transform_test(img_grey).cuda()
    img = transform_test(img_grey)
    img = torch.unsqueeze(img, 0)
    images = img.to(device)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    return predictions.item()
    
# app = Flask(__name__)
app = Flask(__name__, static_url_path = "", static_folder = "static")

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
    
@app.route('/predict',methods = ['GET', 'POST'])
# @app.route('/predict',methods = ['POST'])
def predict():
    # Get the file from post request
    f = request.files['file']

    basepath = os.path.dirname(os.path.realpath('__file__'))
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Make prediction
    preds = model_predict(file_path, model)
    return str(preds)

if __name__ == "__main__":
    app.run(debug=True)