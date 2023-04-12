import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print("Using %s device"%(device))

class panel_nn(nn.Module):
    def __init__(self):
        super(panel_nn,self).__init__()
        self.network = nn.Sequential(
                nn.Conv2d(1,1,kernel_size=3,padding=1),
                nn.ReLU(),
                #nn.MaxPool2d(2,stride=2),
                nn.Flatten(),
                nn.Linear(128**2,32),
                nn.ReLU(),
                nn.Linear(32,32),
                nn.ReLU(),
                nn.Linear(32,32),
                nn.ReLU(),
                nn.Linear(32,9),
            )
        self.flatten = nn.Flatten()
        
    def forward(self,x):
        logits = self.network(x)
        return logits

def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    epoch_loss = 0
    for batch, (inputs,label) in enumerate(dataloader):
        inputs,label = inputs.to(device), label.to(device)
        
        prediction = model(inputs)
        loss = loss_function(prediction, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss+=loss.item()
        if batch%(num_batches//4) == 0:
            lossnum, current = loss.item(), batch*len(inputs)
            print("loss: %.4f at %d/%d"%(lossnum,current,size))
    epoch_loss/=len(dataloader)
    print("epoch loss: %.4f"%(epoch_loss))
            
def test(dataloader, model, loss_function, scheduler=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for inputs,label in dataloader:
            inputs,label = inputs.to(device),label.to(device)
            prediction = model(inputs)
            test_loss += loss_function(prediction,label).item()
            correct += (prediction.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    if scheduler:
        scheduler.step(test_loss)
    correct /= size
    print("test error:")
    print("\taccuracy: %.1f"%(100*correct))
    print("\tavg loss: %.4f"%(test_loss))

def test_confusion(dataloader, model, loss_function):
    label_list = [int(label) for label in dataloader.dataset.classes]
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    cnf_matrix = np.zeros((len(label_list),len(label_list)))
    with torch.no_grad():
        for inputs,label in dataloader:
            inputs,label = inputs.to(device),label.to(device)
            prediction = model(inputs)
            #test_loss += loss_function(prediction,label).item()
            #correct += (prediction.argmax(1) == label).type(torch.float).sum().item()
            label_array = label.cpu().detach().numpy()
            pred_array = prediction.argmax(1).cpu().detach().numpy()
            #print(label_array)
            #print(pred_array)
            new_cnf = confusion_matrix(label_array,pred_array,labels=range(len(label_list)))
            cnf_matrix += new_cnf
    #test_loss /= num_batches
    #scheduler.step(test_loss)
    #correct /= size
    print(cnf_matrix)
    print(dataloader.dataset.class_to_idx)
    return cnf_matrix

train_transform = transforms.Compose([transforms.Grayscale(),
                                      #transforms.Resize(64),
                                      ToTensor()])
train_dataset = datasets.ImageFolder("traincomics",transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#print(len(train_dataset))

for X, y in train_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

test_transform = transforms.Compose([transforms.Grayscale(),
                                      #transforms.Resize(64),
                                      ToTensor()])
test_dataset = datasets.ImageFolder("testcomics",transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = panel_nn().to(device)
loss_function = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), 
#                            lr=5e-3, 
#                            weight_decay=0.05
#                            )
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-2,
                             betas=(0.9,0.999),
                             eps=1e-08,
                             weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       patience=2,
                                                       factor=0.5,
                                                       verbose=True)

epochs = 20
for t in range(epochs):
    print("-------------------- Epoch %d --------------------"%(t))
    train(train_dataloader,model,loss_function,optimizer)
    test(test_dataloader,model,loss_function,scheduler)
    
cnf_matrix = test_confusion(test_dataloader,model,loss_function)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix/10,
                              display_labels=test_dataset.classes)
disp.plot()
plt.show()

torch.save(model.state_dict(), "./panel_counter_model.pth")