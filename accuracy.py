import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import sys
from tqdm import tqdm

def get_data(test_type):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    data_root = None
    if test_type == 'combined':
        data_root='./final_combined/test/'
    elif test_type == 'damagenet':
        data_root='./final_damagenet/test/'
    else:
        data_root='./final_imagenette/test/'
    test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader

def get_model(model_path):
    # Load the model class
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(nn.Flatten(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.Softmax(0))

    # Load the model weights and state from the input path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def run_accuracy_test(test_loader, model):
    model.eval()
    correct_preds_counter = 0
    counter = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            correct_preds_counter += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_acc = 100. * (correct_preds_counter / len(test_loader.dataset))
    print(f'Test accuracy of the model: {epoch_acc} %')

if __name__=="__main__": 
    # Setup input arguments
    parser = argparse.ArgumentParser(
        prog="Applied ML Cloud Final Project Accuracy Tester",
        description="Get the accuracy of the previously a previously trained model on the specified test dataset"
    )
    parser.add_argument('-m', '--model_filepath', type=str, default='./outputs/combined_model.pth', help="Filepath of previously trained pytorch model to run")
    parser.add_argument('-t', '--test_type', type=str, default='combined', help="Which test dataset to run on")
    args = parser.parse_args()
    if args.test_type == 'combined':
        print("Running tests on the combined Imagenette and DamageNet dataset")
    elif args.test_type == 'damagenet':
        print("Running tests on the DamageNet dataset")
    elif args.test_type == 'imagenette':
        print("Running tests on the Imagenette dataset")
    else:
        print("Invalid input, please input valid test type (\"combined\", \"damagenet\", or \"imagenette\")")
        sys.exit(1)

    # Setup cuda device and show whether CPU or GPU is being used
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('Running tests on GPU')
    else:
        print('Running tests on CPU')

    # Setup the test data and retreive the model
    test_loader = get_data(args.test_type)
    model = get_model(args.model_filepath)

    # Run the accuracy test
    run_accuracy_test(test_loader, model)