import numpy as np
from utils import *
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import ipdb

timewin_images = np.load('time_win.npy') # (7, 2670, 3, 32, 32)
av_feats = np.load('avg.npy') # (2670, 3, 32, 32)
target = np.load('target.npy') # (2670, )
fold_pairs = np.load('fold_pairs.npy') # (13, 2)
# ipdb.set_trace()

print ('==> Done loading npy files')

(X_train,y_train),(X_validation,y_validation),(X_test,y_test) = subject_based_train_test_splite(timewin_images,target,fold_pairs[2],option='2sets')

train_dataset = dataset(imgs=X_train,target=y_train)
test_dataset = dataset(imgs=X_test,target=y_test)
train_loader = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True,num_workers=4)
test_loader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=False,num_workers=4)

vgg = models.vgg19(pretrained = True)
model = MyVGG(vgg)

model_lstm = LSTM_conv(embedding=512,hidden=64,net=model)

CUDA = torch.cuda.is_available()
if CUDA:
	model.cuda()
	model_lstm.cuda()

print ('==> Done loading model')

cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss()
if CUDA:
	criterion.cuda()

optimizer_lstm = torch.optim.Adam(model_lstm.parameters(),lr=1e-4,weight_decay=1e-8)
epochs =100

for epoch in range(epochs):
    for i, (img, label) in enumerate(train_loader):
        if CUDA:
        	images = torch.autograd.Variable(img).cuda()
        	labels = torch.autograd.Variable(label.squeeze()).cuda()
        else:
        	images = torch.autograd.Variable(img) # ([16, 7, 3, 32, 32])
        	labels = torch.autograd.Variable(label.squeeze()) # ([16])
        
        optimizer_lstm.zero_grad()
        ipdb.set_trace()
        output= model_lstm(images)
        loss = criterion(output,labels)
        loss.backward()
        ec = torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), 5)
        optimizer_lstm.step()
    acc_train = val(train_loader,net_lstm)
    acc_test = val(test_loader,net_lstm)
    print(acc_train,acc_test)


