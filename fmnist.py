from __future__ import print_function
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 5;
batch_size = 100;
learning_rate = 0.001;


#Data Loading
class FashionMNISTDataset(Dataset):
    '''Fashion MNIST Dataset'''
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file
            transform (callable): Optional transform to apply to sample
        """
        
        data = pd.read_csv(csv_file);
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28)#.astype(float);
        self.Y = np.array(data.iloc[:, 0]);
        
        del data;
        self.transform = transform;
        
    def __len__(self):
        return len(self.X);
    
    def __getitem__(self, idx):
        item = self.X[idx];
        label = self.Y[idx];
        
        if self.transform:
            item = self.transform(item);
        
        return (item, label);


# data readers   
train_dataset = FashionMNISTDataset(csv_file='./data/fmnist/fashion-mnist_train.csv');
test_dataset = FashionMNISTDataset(csv_file='./data/fmnist/fashion-mnist_test.csv')    

#define the data loaders using the DataLoader module
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True);
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True);
                                          
                                          
#We can also use the data reader to visualize some images and see the dataset                                          
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};
fig = plt.figure(figsize=(8,8));
columns = 4;
rows = 5;
for i in range(1, columns*rows +1):
    img_xy = np.random.randint(len(train_dataset));
    img = train_dataset[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()                                          
    
#model structure
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Sequential( nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,bias=True),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2,bias=True),
                                   nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2)
                                   )
        
        self.out = nn.Linear(in_features=32*7*7, out_features=10)
        
        
    def forward(self,t):
        out = self.conv1(t)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.out(out)
        return out
 
    
#instance od our model       
network= Network()       
#declare the loss function to optimize       
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate);

# training our model
losses = [];
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = network(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.data);
        
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data))

#evaluate it on our test dataset
network.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.float())
    outputs = network(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))


