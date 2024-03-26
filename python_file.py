# making the changes as mentioned in the experiment branch 
# adding the code of class assignment  2 q1 
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

defining the required MLP model

# Modify MLP model architecture
class MY_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MY_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Adding another hidden layer
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Adjusting output layer

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Apply activation function for the new layer
        x = self.fc3(x)
        return x

defining the required CNN Architectre for solving the problem



class CNN_Architecture(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Architecture, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Updating the input size here
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

defining the function for evaluating the model

def evaluating_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            targets.extend(labels.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')
    confusion_mat = confusion_matrix(targets, predictions)
    return accuracy, precision, recall, confusion_mat

defining the data transformations

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


now loading the necessary USPS data

training_dataset = datasets.USPS(root='./data', train=True, transform=transform, download=True)
testing_dataset = datasets.USPS(root='./data', train=False, transform=transform, download=True)

now defining the data loaders

training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
testing_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False)

intializing our model

input_size = 16 * 16
hidden_size = 128
num_classes = 10
my_mlp_model = MY_MLP(input_size, hidden_size, num_classes)
my_cnn_model = CNN_Architecture(num_classes)

defining loss function and optimizers

criterion = nn.CrossEntropyLoss()
# changing the learning rate for optimizers

my_mlp_optimizer = optim.Adam(my_mlp_model.parameters(), lr=0.0005)
my_cnn_optimizer = optim.Adam(my_cnn_model.parameters(), lr=0.0005)


intializing tensorboard writer


writer = SummaryWriter()

defining epochs

epochs = 10

training mlp model


for epoch in range(epochs):
    my_mlp_model.train()
    for images, labels in training_loader:
        my_mlp_optimizer.zero_grad()
        outputs = my_mlp_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        my_mlp_optimizer.step()
    # Log loss to TensorBoard
    writer.add_scalar('MLP/loss', loss, epoch)

training CNN model

for epoch in range(epochs):
    my_cnn_model.train()
    for images, labels in training_loader:
        my_cnn_optimizer.zero_grad()
        outputs = my_cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        my_cnn_optimizer.step()
    # Log loss to TensorBoard
    writer.add_scalar('CNN/loss', loss, epoch)

Evaluating the model

my_mlp_accuracy, my_mlp_precision, my_mlp_recall, my_mlp_confusion_mat = evaluating_model(my_mlp_model, testing_loader)
my_cnn_accuracy, my_cnn_precision, my_cnn_recall, my_cnn_confusion_mat = evaluating_model(my_cnn_model, testing_loader)



writer.close()

displaying the results

print("MLP Accuracy:", my_mlp_accuracy)
print("MLP Precision:", my_mlp_precision)
print("MLP Recall:", my_mlp_recall)
print("MLP Confusion Matrix:\n", my_mlp_confusion_mat)

print("CNN Accuracy:", my_cnn_accuracy)
print("CNN Precision:", my_cnn_precision)
print("CNN Recall:", my_cnn_recall)
print("CNN Confusion Matrix:\n", my_cnn_confusion_mat)
