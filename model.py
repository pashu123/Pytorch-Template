import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os 
import numpy as np
from trainloop import RunManager,RunBuilder
from collections import OrderedDict

## This is cpu version for gpu we have to change the variable names and type
## Data augmentation and normalization for training

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406] , [0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406] , [0.229,0.224,0.225])
    ])
}


## Specify the data directory
data_dir = 'hymenoptera_data'

## Provide the train and validation set for the training images
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}



# Load the Different models 
model_conv = torchvision.models.mobilenet_v2(pretrained = True)

# print(model_conv.parameters)

# Freeze all layers in the network or non freeze and train from scratch
for param in model_conv.parameters():
    param.requires_grad = False

model_conv.classifier = nn.Linear(in_features = 1280,out_features = 2,bias= True)





# if torch.cuda.is_available():
#     print('Cuda is available')
#     model_conv =  model_conv.cuda()


## Lets understand what's happening

criterion = nn.CrossEntropyLoss()

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [10, 20],
    shuffle = [True]
)



## Set the number of epochs
num_epochs = 10


m = RunManager()

for run in RunBuilder.get_runs(params):

    network = model_conv
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x] , batch_size = run.batch_size,  
                                                    shuffle = run.shuffle) for x in ['train','val']}

    optimizer = optim.SGD(network.parameters(),lr = run.lr,momentum= 0.9)


    m.begin_run(run,network,dataloaders['train'])

    for epoch in range(num_epochs):
        m.begin_epoch()
       
        
        for images,labels in dataloaders['train']:
            images = Variable(images)
            labels = Variable(labels)

            # if torch.cuda.is_available():
            #     images = images.cuda()
            #     labels = labels.cuda()

            preds = network(images)
            loss = criterion(preds,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## Send the loss and preds labels to manager
            m.track_loss(loss)
            m.track_num_correct(preds,labels)

        m.end_epoch()

    m.end_run()

    m.save_results('results')
    m.save_model('model')
      





