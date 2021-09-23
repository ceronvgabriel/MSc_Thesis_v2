import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import torchvision
from torchvision import models
import torchvision.transforms as transforms

import log
from models_c import *
#from utils import progress_bar
from utils import progress_bar_none as progress_bar
#from utils import progress_bar_print as progress_bar
import os
import inspect
import time


t_batch_size=1024 # Change this value when needed, also num workers
n_workers=6

test_batch_size=128

# Import as module global variables so we don't have to execute it every time
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch_size, shuffle=False, num_workers=n_workers)

#Train:
classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Data

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=t_batch_size, shuffle=True, num_workers=n_workers)


#Function to test a given model
def test(model):
    #model.eval()
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #print('Loss: %.3f | Acc: %.3f%% (%d/%d)'%(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return correct/total , test_loss


def load(save_folder,num_classes=10):
    '''saved_folder, num_classes=10'''

    os.chmod(save_folder,777)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = models.resnet18(num_classes=num_classes)
    #net = ResNet18()
    net = net.to(device)

    if device == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load(save_folder + '/ckpt.pth',map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])
    return net

def load_meta(save_folder,num_classes=10):
    '''Load metadata of trained model,saved state is of the form:
    state = {
        'net': net.state_dict(),
        'train_acc':train_acc_v[-1],
        'test_acc': test_acc_v[-1],
        'train_loss' : train_loss_v[-1],
        'test_loss' : test_loss_v[-1],
        'epoch': epoch,
        'best_acc': best_acc,
        # Next lines are done below
        #'current_lr': scheduler.get_last_lr(),
        #'current_lr': optimizer.param_groups[0]['lr'],
        'parameters': parameters,

    }
    '''

    os.chmod(save_folder,777)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        cudnn.benchmark = True

    checkpoint = torch.load(save_folder + '/ckpt.pth',map_location=torch.device(device))
    return checkpoint

#load first custom net from git, the first one you used
def load_custom(save_folder,num_classes=10):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #net = models.resnet18(num_classes=num_classes)
    net = ResNet18()
    
    net = net.to(device)

    if device == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load('./checkpoints/' + save_folder + '/ckpt.pth',map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])
    return net


#Training

#Train model with CIFAR
def train_model(model,epochs,save_folder,t_batch_size=128):
    
    print("Training")
    #global args
    best_acc=0
    start_epoch=0


    net = model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device='cpu'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, #MOD: lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        nonlocal best_acc
        nonlocal save_folder
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        #Early stop to prevent overfitting
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints/'+save_folder):
                os.makedirs('checkpoints/'+save_folder)
            torch.save(state, './checkpoints/'+save_folder+'/ckpt.pth')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
    return best_acc


def train_model_resume(model,epochs,save_folder,resume_folder=None,t_batch_size=128):
    '''If no resume_folder is passed, it trains from zero'''
    
    print("Training")
    #global args
    best_acc=0
    start_epoch=0


    # Model

    net = model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device='cpu'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, #MOD: lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if resume_folder is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoints/'+resume_folder), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoints/'+resume_folder+'/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        nonlocal best_acc
        nonlocal save_folder
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        #Early stop to prevent overfitting
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc or epoch==(epochs-1):
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints/'+save_folder):
                os.makedirs('checkpoints/'+save_folder)
            torch.save(state, './checkpoints/'+save_folder+'/ckpt.pth')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
    return best_acc


def train_model_resume_noEarly(model,epochs,save_folder,resume_folder=None,t_batch_size=128):
    '''If no resume_folder is passed, it trains from zero'''
    
    print("Training\n")
    #global args
    best_acc=0
    start_epoch=1

    train_acc_v=[]
    test_acc_v=[]

    train_loss_v=[]
    test_loss_v=[]


    # Model

    net = model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device='cpu'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, #MOD: lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if resume_folder is not None:
        # Load checkpoint.
        print('Resuming..')
        assert os.path.isdir('checkpoints/'+resume_folder), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoints/'+resume_folder+'/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        print("Resumed epoch: " + str(start_epoch) +"\n")

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        train_acc=100.*correct/total
        train_acc_v.append(train_acc)

        train_loss_avg=train_loss/trainloader.batch_size
        train_loss_v.append(train_loss_avg)


    def test(epoch):
        nonlocal best_acc
        nonlocal save_folder
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            test_acc=100.*correct/total
            test_acc_v.append(test_acc)
    
            test_loss_avg=test_loss/testloader.batch_size
            test_loss_v.append(test_loss_avg)

        acc = 100.*correct/total
        if acc>best_acc:
            best_acc = acc

        #Save only the last epoch
        if epoch == (start_epoch+epochs-1):
            print('Saving.. epoch: ' + str(epoch) +"\n")
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints/'+save_folder):
                os.makedirs('checkpoints/'+save_folder)
            torch.save(state, './checkpoints/'+save_folder+'/ckpt.pth')


    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
    
    res_dict={}
    for i in ["train_acc_v","train_loss_v","test_acc_v","test_loss_v"]:
        res_dict[i]=eval(i)


    return res_dict

def progressive_train_1(step,total,name):
    '''step,total(number of steps),name(checkpoints save folder name)'''
    model=models.resnet18(num_classes=10)
    name_i=name+"_epoch_"+str(step)
    name_i= name + "/" + name_i # create subfolder
    res_v={}

    res=log.log_save(name_i,model_actions.train_model_resume_noEarly,model,epochs=step,save_folder=name_i)
    #res=log.log_save_(model_actions.train_model_resume,name_i,model,epochs=step,save_folder=name_i)
    res_v=res
    print(res["test_acc_v"][-1])
    log.log("Epoch: "+str(step) + " Accuracy: " + str(res["test_acc_v"][-1]) + " Loss: " + str(res["test_loss_v"][-1]))

    for i in range(2,total+1):# 2 because of previos train

        name_i=name+"_epoch_"+str(step*i) #next name
        name_j=name+"_epoch_"+str(step*i - step) #prev name

        name_i= name + "/" + name_i # Create subfolder
        name_j= name + "/" + name_j

        res=log.log_save(name_i,model_actions.train_model_resume_noEarly,model,epochs=step,save_folder=name_i,resume_folder=name_j)
        print(res["test_acc_v"][-1])
        log.log("Epoch: "+str(step*i) + " Accuracy: " + str(res["test_acc_v"][-1]) + " Loss: " + str(res["test_loss_v"][-1]))

        #append all results
        res_v["test_acc_v"].extend(res["test_acc_v"])
        res_v["test_loss_v"].extend(res["test_loss_v"])
        res_v["train_acc_v"].extend(res["train_acc_v"])
        res_v["train_loss_v"].extend(res["train_loss_v"])

    log.save(res_v,name)
    
    return res_v


def progressive_train_2(model,epochs,save_folder,step=None):
    '''If no resume_folder is passed, it trains from zero'''

    if step == None:
        step=epochs
    
    print("Training\n")
    #global args
    best_acc=0
    start_epoch=1

    train_acc_v=[]
    test_acc_v=[]

    train_loss_v=[]
    test_loss_v=[]


    # Model

    net = model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device='cpu'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, #MOD: lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        train_acc=100.*correct/total
        train_acc_v.append(train_acc)

        train_loss_avg=train_loss/trainloader.batch_size
        train_loss_v.append(train_loss_avg)


    def test(epoch):
        nonlocal best_acc
        nonlocal save_folder
        #nonlocal step
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            test_acc=100.*correct/total
            test_acc_v.append(test_acc)
    
            test_loss_avg=test_loss/testloader.batch_size
            test_loss_v.append(test_loss_avg)
        best_acc_flag=0
        acc = 100.*correct/total
        if acc>best_acc:
            best_acc = acc
            best_acc_flag=1

        #Save only models for the given step
        if epoch%step ==0 or acc > best_acc:
            print('Saving.. epoch: ' + str(epoch) +"\n")
            state = {
                'net': net.state_dict(),
                'train_acc':train_acc_v[-1],
                'test_acc': test_acc_v[-1],
                'train_loss' : train_loss_v[-1],
                'test_loss' : test_loss_v[-1],
                'epoch': epoch,
                'current_lr': scheduler.get_last_lr()
            }
            if not os.path.isdir('checkpoints/'+save_folder+"_epoch_"+str(epoch)):
                os.makedirs('checkpoints/'+save_folder+"_epoch_"+str(epoch),777)
            if best_acc_flag:
                torch.save(state, './checkpoints/'+save_folder+"_best_acc"+'/ckpt.pth')
            else:
                torch.save(state, './checkpoints/'+save_folder+"_epoch_"+str(epoch)+'/ckpt.pth')

    #driver:
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
    
    res_dict={}
    for i in ["train_acc_v","train_loss_v","test_acc_v","test_loss_v"]:
        res_dict[i]=eval(i)

    log.save(res_dict,save_folder.split("/")[-1])
    return res_dict

def progressive_train_3(model,epochs,save_folder,step=None,criterion=None, scheduler=None):
    '''Progressive training of a model, save model state every step until number of epochs is reached'''

    if step == None:
        step=epochs

    dataloaders = {"train":trainloader, "test": testloader}
    datasets = {"train": trainset , "test":testset}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}
    
    print("Training\n")
    #global args
    best_acc=0
    start_epoch=1

    train_acc_v=[]
    test_acc_v=[]

    train_loss_v=[]
    test_loss_v=[]
    current_lr_v=[]


    # Model

    net = model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device='cpu'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if scheduler is None:
        optimizer = optim.SGD(net.parameters(), lr=0.1, #MOD: lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        optimizer=scheduler.optimizer
    
    current_f=str(inspect.currentframe().f_code.co_name)
    
    parameters={"current_function":current_f,"batch_size":t_batch_size,"n_workers":n_workers,"optimizer":{"class":optimizer.__class__,"dict":optimizer.defaults},"scheduler":{"class":scheduler.__class__,"dict":scheduler.__dict__}}

    start = time.process_time()
    current = time.process_time() - start

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() *inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        train_acc=100.*correct/total
        train_acc_v.append(train_acc)

        train_loss_avg= train_loss / dataset_sizes["train"]
        train_loss_v.append(train_loss_avg)


    def test(epoch):
        nonlocal best_acc
        nonlocal save_folder
        #nonlocal step
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()*inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            test_acc=100.*correct/total
            test_acc_v.append(test_acc)
    
            test_loss_avg= test_loss / dataset_sizes["test"]
            test_loss_v.append(test_loss_avg)
        best_acc_flag=0
        acc = 100.*correct/total
        if acc>best_acc:
            best_acc = acc
            best_acc_flag=1

        #Save only models for the given step
        if epoch%step ==0 or acc > best_acc:
            print('Saving.. epoch: ' + str(epoch) +"\n")
            state = {
                'net': net.state_dict(),
                'train_acc':train_acc_v[-1],
                'test_acc': test_acc_v[-1],
                'train_loss' : train_loss_v[-1],
                'test_loss' : test_loss_v[-1],
                'epoch': epoch,
                'best_acc': best_acc,
                # Next lines are done below
                #'current_lr': scheduler.get_last_lr(),
                #'current_lr': optimizer.param_groups[0]['lr'],
                'parameters': parameters,

            }
            if not os.path.isdir('checkpoints/'+save_folder+"_epoch_"+str(epoch)):
                os.makedirs('checkpoints/'+save_folder+"_epoch_"+str(epoch),777)
            if not os.path.isdir('checkpoints/'+save_folder+"_best_acc"):
                os.makedirs('checkpoints/'+save_folder+"_best_acc",777)
            if best_acc_flag:
                torch.save(state, './checkpoints/'+save_folder+"_best_acc"+'/ckpt.pth')
            
            torch.save(state, './checkpoints/'+save_folder+"_epoch_"+str(epoch)+'/ckpt.pth')

    #driver:
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        test(epoch)

        if scheduler.__class__.__name__== 'CosineAnnealingLR':
            scheduler.step()
            current_lr_v.append(scheduler.get_last_lr())
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(test_loss_v[-1])
            current_lr_v.append(optimizer.param_groups[0]['lr'])

        current = time.process_time() - start

    #save and return training meta data
    res_dict={}
    for i in ["train_acc_v","train_loss_v","test_acc_v","test_loss_v","current_lr_v"]:
        res_dict[i]=eval(i)

    res_dict["parameters"]=str(eval("parameters"))
    res_dict["tot_time"]=current

    log.save(res_dict,save_folder.split("/")[-1])
    return res_dict

def progressive_train_4(model,epochs,step=None,tr_bs=128,save_folder=None,criterion=None, scheduler=None):
    '''Like progressive 3 but with modifable training bs, save_folder must be of the form: "test_progressive3/test_progressive3"'''

    if step == None:
        step=epochs
    
    #Setting training batch size
    trainloader_c = torch.utils.data.DataLoader(
    trainset, batch_size=tr_bs, shuffle=True, num_workers=n_workers)

    #just to get dataset sizes
    dataloaders = {"train":trainloader_c, "test": testloader}
    datasets = {"train": trainset , "test":testset}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}
    
    print("Training\n")
    #global args (outer)
    best_acc=0
    start_epoch=1

    train_acc_v=[]
    test_acc_v=[]

    train_loss_v=[]
    test_loss_v=[]
    current_lr_v=[]


    # Model

    net = model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device='cpu'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if scheduler is None:
        optimizer = optim.SGD(net.parameters(), lr=0.1, #MOD: lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        optimizer=scheduler.optimizer
    
    #get current function
    current_f=str(inspect.currentframe().f_code.co_name)
    
    #hyperparameters to save
    parameters={"current_function":current_f,"batch_size":t_batch_size,"n_workers":n_workers,"optimizer":{"class":optimizer.__class__,"dict":optimizer.defaults},"scheduler":{"class":scheduler.__class__,"dict":scheduler.__dict__}}

    start = time.process_time()
    current = time.process_time() - start

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader_c):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() *inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(trainloader_c), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        train_acc=100.*correct/total
        train_acc_v.append(train_acc)

        train_loss_avg= train_loss / dataset_sizes["train"]
        train_loss_v.append(train_loss_avg)


    def test(epoch):
        nonlocal best_acc
        nonlocal save_folder
        #nonlocal step
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()*inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            test_acc=100.*correct/total
            test_acc_v.append(test_acc)
    
            test_loss_avg= test_loss / dataset_sizes["test"]
            test_loss_v.append(test_loss_avg)
        
        acc = 100.*correct/total

        #Check if it is the best acc up to now
        best_acc_flag=0
        if acc>best_acc:
            best_acc = acc
            best_acc_flag=1

        #Save only models for the given step or best acc
        if epoch%step ==0 or best_acc_flag==1:
            print('Saving.. epoch: ' + str(epoch) +"\n")
            state = {
                'net': net.state_dict(),
                'train_acc':train_acc_v[-1],
                'test_acc': test_acc_v[-1],
                'train_loss' : train_loss_v[-1],
                'test_loss' : test_loss_v[-1],
                'epoch': epoch,
                'best_acc': best_acc,
                'parameters': parameters,

            }
            if not os.path.isdir('checkpoints/'+save_folder+"_epoch_"+str(epoch)):
                os.makedirs('checkpoints/'+save_folder+"_epoch_"+str(epoch),777)
            if not os.path.isdir('checkpoints/'+save_folder+"_best_acc"):
                os.makedirs('checkpoints/'+save_folder+"_best_acc",777)
            if best_acc_flag:
                torch.save(state, './checkpoints/'+save_folder+"_best_acc"+'/ckpt.pth')
            
            torch.save(state, './checkpoints/'+save_folder+"_epoch_"+str(epoch)+'/ckpt.pth')

    #driver:
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        test(epoch)

        if scheduler.__class__.__name__== 'CosineAnnealingLR':
            scheduler.step()
            current_lr_v.append(scheduler.get_last_lr())
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(test_loss_v[-1])
            current_lr_v.append(optimizer.param_groups[0]['lr'])

        current = time.process_time() - start

    #save and return training meta data
    res_dict={}
    for i in ["train_acc_v","train_loss_v","test_acc_v","test_loss_v","current_lr_v"]:
        res_dict[i]=eval(i)

    res_dict["parameters"]=str(eval("parameters"))
    res_dict["tot_time"]=current

    log.save(res_dict,save_folder.split("/")[-1])
    return res_dict


# def train_model_resume_noEarly(model,epochs,save_folder,resume_folder=None,t_batch_size=128):
    
#     print("Training")
#     #global args
#     best_acc=0
#     start_epoch=0

#     # Data
#     trainset = torchvision.datasets.CIFAR10(
#         root='./data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=t_batch_size, shuffle=True, num_workers=1)

#     # Model
#     print('==> Building model..')

#     net = model

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     #device='cpu'
#     net = net.to(device)
#     if device == 'cuda':
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = True
        
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.1, #MOD: lr=args.lr,
#                         momentum=0.9, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#     if resume_folder is not None:
#         # Load checkpoint.
#         print('==> Resuming from checkpoint..')
#         assert os.path.isdir('checkpoints/'+resume_folder), 'Error: no checkpoint directory found!'
#         checkpoint = torch.load('./checkpoints/'+resume_folder+'/ckpt.pth')
#         net.load_state_dict(checkpoint['net'])
#         best_acc = checkpoint['acc']
#         start_epoch = checkpoint['epoch']

#     # Training
#     def train(epoch):
#         print('\nEpoch: %d' % epoch)
#         net.train()
#         train_loss = 0
#         correct = 0
#         total = 0
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


#     def test(epoch):
#         nonlocal best_acc
#         nonlocal save_folder
#         net.eval()
#         test_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch_idx, (inputs, targets) in enumerate(testloader):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = net(inputs)
#                 loss = criterion(outputs, targets)

#                 test_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()

#                 progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#         #no early stop
#         # Save checkpoint.
#         # acc = 100.*correct/total
#         # if acc > best_acc:




#     for epoch in range(start_epoch, start_epoch+epochs):
#         train(epoch)
#         test(epoch)
#         scheduler.step()
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
    
#     #save just at the end:
#     if not os.path.isdir('checkpoints/'+save_folder):
#         os.makedirs('checkpoints/'+save_folder)
#     torch.save(state, './checkpoints/'+save_folder+'/ckpt.pth')
#     best_acc = acc
    
#     return best_acc