import torch
import torch.utils.data as data
import numpy as np
from torchnet.meter import AverageValueMeter
import ipdb

def subject_based_train_test_splite(X,y,folder,option='3sets'):
    if option=='3sets':
        dim = len(X.shape)
        indice_train = folder[0][len(folder[1]):]
        indice_validation = folder[0][:len(folder[1])]
        indice_test = folder[1]
        if dim ==2 or dim==4: # for feats split:
            X_train,y_train = X[indice_train],y[indice_train]
            X_validation, y_validation = X[indice_validation],y[indice_validation]
            X_test, y_test = X[indice_test], y[indice_test]
        if dim ==5:
            X_train,y_train = X[:,indice_train],y[indice_train]
            X_validation, y_validation = X[:,indice_validation],y[indice_validation]
            X_test, y_test = X[:,indice_test], y[indice_test]
        return (X_train,y_train),(X_validation,y_validation),(X_test,y_test)
    else:
        dim = len(X.shape) # 5
        indice_train = folder[0] # (2471, )
        indice_test = folder[1] # (199, )
        if dim == 2 or dim == 4:  # for feats split:
            X_train, y_train = X[indice_train], y[indice_train]
            X_test, y_test = X[indice_test], y[indice_test]
        if dim == 5:
            X_train, y_train = X[:, indice_train], y[indice_train]

            X_test, y_test = X[:, indice_test], y[indice_test]
        return (X_train, y_train), (None, None), (X_test, y_test)

class MyVGG(torch.nn.Module):
    def __init__(self, vgg):
        super(MyVGG, self).__init__()
        self.features = vgg.features#torch.nn.Sequential(*list(vgg.children()))
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(512, 1024),
        #     torch.nn.ReLU(True),
        #     torch.nn.Dropout(),
        #     torch.nn.Linear(1024, 1024),
        #     torch.nn.ReLU(True),
        #     torch.nn.Dropout(),
        #     torch.nn.Linear(1024, num_classes))
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

class dataset(data.Dataset):
    def __init__(self,imgs,target,transform=None):
        super().__init__()

        np.random.seed(1)
        self.imgs =imgs
        self.target = target

    def __getitem__(self, index):
        if len(self.imgs.shape)==4:
            image = torch.FloatTensor(self.imgs[index])
            target = torch.LongTensor([int(self.target[index])])
            return image,target
        if len(self.imgs.shape)==5:
            image = torch.FloatTensor(self.imgs[:,index])
            target = torch.LongTensor([int(self.target[index])])
            return image,target

    def __len__(self):
        if len(self.imgs.shape)==4:
            return len(self.imgs)
        elif len(self.imgs.shape)==5:
            return self.imgs.shape[1]

class LSTM_conv(torch.nn.Module):
    def __init__(self,embedding=512,hidden=64,net=None):
        super().__init__()
        self.net = net
        self.embedding = embedding
        self.hidden = hidden
        self.rnn = torch.nn.LSTM(input_size=embedding,hidden_size=hidden,batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=960,out_features=200),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(200,4)
            )
        self.conv = torch.nn.Conv1d(in_channels=7,out_channels=1,kernel_size=1)
    def forward(self, input):
        shapes = input.shape # ([16, 7, 3, 32, 32])
        features = self.net(input.view(shapes[0]*shapes[1],shapes[2],shapes[3],shapes[4])) # ([112, 512, 1, 1])
        features_reshape = features.view(shapes[0],shapes[1],512) # ([16, 7, 512])
        output_cnn = self.conv(features_reshape).squeeze() # ([16, 512])

        output, hidden = self.rnn(features_reshape) # output: ([16, 7 ,64]), hidden: [([1, 16, 64]), ([1, 16, 64])]

        output = self.classifier(torch.cat([output.contiguous().view(16, -1), output_cnn], 1))
        return output # ([16, 4])

def val(dataloader,net):
    net.eval()
    acc = AverageValueMeter()
    acc.reset()
    for i, (img, label) in enumerate(dataloader):
        batch_size = len(label)
        images = Variable(img).cuda()
        labels = Variable(label.squeeze()).cuda()
        output = net(images)
        predictedLabel = torch.max(output,1)[1]
        acc_ = (predictedLabel==labels).sum().type(torch.FloatTensor)/batch_size
        acc.add(acc_.item())
    net.train()
    return acc.value()[0]
