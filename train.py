import torch
import torch.nn.functional as F

import os
from utils import AverageMeter, calculate_accuracy, calculate_class_num
from plot_results import plot_confusion_matrix, plot_roc_curve


def train(model, criterion, optimizer, train_loader, test_loader, num_epochs, device, cfg, label_dict):

    model.train()

    for epoch in range(1, num_epochs+1):

        losses = AverageMeter()
        accuracies = AverageMeter()

        print(f'Starting training on epoch {epoch} / {num_epochs}')

        all_targets_list = []

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            all_targets_list.append(targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 30 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i + 1, len(train_loader), loss=losses, acc=accuracies))
        
        all_targets = torch.cat(all_targets_list, dim=0).tolist()
        calculate_class_num(all_targets, label_dict, 'train')

        if epoch % 5 == 0:
            if not os.path.exists(cfg.TRAIN.CHECKPOINT_PATH):
                os.mkdir(cfg.TRAIN.CHECKPOINT_PATH)
            save_file_path = os.path.join(cfg.TRAIN.CHECKPOINT_PATH, 'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        
        print('Epoch: {0}\t'
              'Train Loss {1}\t'
              'Train Acc {2}'.format(epoch, losses.avg, accuracies.avg))
              
        test(model, test_loader, device, label_dict)


def val(epoch, model, criterion, dataloader, device):

    print('validation at epoch {}'.format(epoch))

    losses = AverageMeter()
    accuracies = AverageMeter()

    model.eval()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

    print('Epoch: {0}\t'
          'Val Loss {1}\t'
          'Val Acc {2}'.format(epoch, losses.avg, accuracies.avg))


def test(model, dataloader, device, label_dict):
    accuracies = AverageMeter()
    accuracies_top3 = AverageMeter()

    model.eval()
    all_targets_list = []
    all_outputs_list = []

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            outputs = model(inputs)
            acc = calculate_accuracy(outputs, targets)
            acc3 = calculate_accuracy(outputs, targets, topk=3)
            accuracies.update(acc, inputs.size(0))
            accuracies_top3.update(acc3, inputs.size(0))
            all_targets_list.append(targets)
            all_outputs_list.append(outputs)
    
    all_targets = torch.cat(all_targets_list, dim=0).tolist()
    all_outputs = torch.argmax(torch.cat(all_outputs_list, dim=0), dim=1).tolist()
    calculate_class_num(all_targets, label_dict, 'test')
    plot_confusion_matrix(all_targets, all_outputs, label_dict)
    plot_roc_curve(all_targets, all_outputs, label_dict)
    
    print(f'Test Top1 Acc: {accuracies.avg},  Test Top3 Acc: {accuracies_top3.avg}')
    print('------------------------------------------------------------')
