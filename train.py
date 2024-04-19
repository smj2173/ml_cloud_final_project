import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_data(damaged, batch_size):
    # the training transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    # the validation transforms
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    data_root = './final_combined/' if damaged else './final_imagenette/'
    train_data_root = data_root + 'train/'
    val_data_root = data_root + 'val/'
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_root, transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(root=val_data_root, transform=valid_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, valid_loader

def get_model():
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(nn.Flatten(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    return model.to(device), loss_fn, optimizer

# training
def train(model, train_loader, optimizer, loss_fn):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_fn(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc

# validation
def validate(model, valid_loader, loss_fn):
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = loss_fn(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))
    return epoch_loss, epoch_acc

def save_plots(train_accuracy, valid_accuracy, train_loss, valid_loss, damaged):
    # accuracy plots
    accuracy_plot_path = './outputs/combined_accuracy.png' if damaged else './outputs/imagenette_accuracy.png'
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='red', linestyle='-', label='train accuracy')
    plt.plot(valid_accuracy, color='blue', linestyle='-', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(accuracy_plot_path)
    
    # loss plots
    loss_plot_path = './outputs/combined_loss.png' if damaged else './outputs/imagenette_loss.png'
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='red', label='train loss')
    plt.plot(valid_loss, color='blue', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)

def save_model(num_epochs, model, optimizer, loss_fn, damaged):
    output_location = './outputs/combined_model.pth' if damaged else './outputs/imagenette_model.pth' 
    torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                }, output_location)

if __name__=="__main__": 
    # Setup input arguments
    parser = argparse.ArgumentParser(
        prog="Applied ML Cloud Final Project Trainer",
        description="Trains a Resnet18 model on Imagenette data by default, or combined Imagenette and DamageNet data using --damaged"
    )
    parser.add_argument('-d', '--damaged', action='store_true', help="Include the damaged data in training")
    parser.add_argument('-c', '--cpu', action='store_true', help="Use cpu to do training")
    parser.add_argument('-n', '--num_epochs', type=int, default=10, help="Number of epochs to run")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size")
    args = parser.parse_args()
    if args.damaged:
        print("Running training on the combined Imagenette and DamagedNet dataset")
    else:
        print("Running training on the Imagenette dataset")

    # Setup cuda device and show whether CPU or GPU is being used
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    if device == 'cuda':
        print('Running training on GPU')
    else:
        print('Running training on CPU')

    # Setup the input data (training and validation) and the final model
    train_loader, valid_loader = get_data(args.damaged, args.batch_size)
    model, loss_fn, optimizer = get_model()

    # Iterate for the given number of epochs
    print("Saving all losses and accuracies for each epoch")
    train_loss, valid_loss = [], []
    train_accuracy, valid_accuracy = [], []
    for epoch in range(args.num_epochs):
        print(f"[INFO]: Epoch {epoch+1} of {args.num_epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, loss_fn)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  loss_fn)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_accuracy.append(train_epoch_acc)
        valid_accuracy.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        
    # Save the trained model weights and the loss/accuracy plots
    save_model(args.num_epochs, model, optimizer, loss_fn)
    save_plots(train_accuracy, valid_accuracy, train_loss, valid_loss, args.damaged)
    print('Complete!')