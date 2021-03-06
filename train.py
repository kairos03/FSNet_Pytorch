import argparse

from sacred import Experiment
from sacred.observers import MongoObserver

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms


import config
from model import fs_resnet


# sacred experiment
ex = Experiment('FSNet')
ex.observers.append(MongoObserver.create(url=config.MONGO_URI,
                                         db_name=config.MONGO_DB))


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FSNet')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--workers', type=int, default=2, metavar='N',
                        help='workers for dataloader')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--milestone', type=int, nargs='*', default=[150, 225], metavar='M',
                        help='Learning rate Scheduling milestone (default: [150, 225])')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    return args


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # gradient logging
        # for n, m in model.named_modules():
        #     if 'conv' in n:
        #         print(m.FS.grad.mean(), m.FS.grad.min(), m.FS.grad.max())
        optimizer.step()

        # calculate metric
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        # logging at console
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx * len(data) / len(train_loader.dataset), loss/len(data)))

    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    
    # logging at sacred
    ex.log_scalar('train.loss', train_loss, epoch)
    ex.log_scalar('train.acc', accuracy, epoch)


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # calculate metric
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # logging at sacred
    ex.log_scalar('test.loss', test_loss, epoch)
    ex.log_scalar('test.acc', accuracy, epoch)

    print('\nTest {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy


@ex.config
def hyperparam():
  """
  sacred exmperiment hyperparams
  :return:
  """
  args = get_params()

  print("hyperparam - ", args)


@ex.main
def main(args, _seed):
    """
    sacred exmperiment main
    :args args: additional aurgment from @ex.config
    :args _seed: random seed from @ex.config
    :return last_acc: experiment result 
    """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(_seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    last_acc = 0

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='/dataset/CIFAR', train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)

    testset = torchvision.datasets.CIFAR10(root='/dataset/CIFAR', train=False,
                                           download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

    model = fs_resnet(data='cifar10', num_layers=20) .to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestone, args.gamma)

    for epoch in range(1, args.epochs + 1):
        with torch.autograd.detect_anomaly():
            train(args, model, device, train_loader, criterion, optimizer, epoch)
        _, last_acc = test(model, device, test_loader, criterion, epoch)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "FSNet_cifar10.pt")
    
    return last_acc


if __name__ == '__main__':
    ex.run()
