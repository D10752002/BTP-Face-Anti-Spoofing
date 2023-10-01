import os
from random import randint
import torch
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AvgMeter
from utils.eval import add_visualization_to_tensorboard, predict, calc_accuracy
from tqdm import tqdm

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    return true_positives, false_positives, true_negatives, false_negatives

#for training the face spoofing model
class FASTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, valloader, testloader, writer):
        super(FASTrainer, self).__init__(cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, valloader, testloader, writer)
        #create network
        self.network = self.network.to(device)
        self.best_val_acc = 0
        #for the metrics
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))

        self.test_loss_metric = AvgMeter(writer=writer, name='Loss/test', num_iter_per_epoch=len(self.testloader))
        self.test_acc_metric = AvgMeter(writer=writer, name='Accuracy/test', num_iter_per_epoch=len(self.testloader))


    def load_model(self, epoch):
        #loading the model, it was changed to have three parameters to load like the save_model
        #to reload the saved type back. originally just had the first two.
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name'], epoch))
        state = torch.load(saved_name)

        print('loaded as', saved_name)
        #loads the model and then returns the loaded state dict which finishes the loading process
        self.optimizer.load_state_dict(state['optimizer'])
        return self.network.load_state_dict(state['state_dict'])

    #it was changed to have three parameters to create a new pth file every time a new epoch is finished
    def save_model(self, epoch):
        
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name'], epoch))
        print('saved as', saved_name)
        #saves all the necessary information
        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(state, saved_name)

    #for training for an epoch, used again and again until epoch number is reached.
    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        for i, (img, depth_map, label) in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
            img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
            net_depth_map = self.network(img)
            self.optimizer.zero_grad()
            loss = self.criterion(net_depth_map, depth_map)
            loss.backward()
            self.optimizer.step()
            #predicting the dpeth map
            preds, _ = predict(net_depth_map)
            #actual depth map
            targets, _ = predict(depth_map)
            # compare the two
            accuracy = calc_accuracy(preds, targets)
            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)
            # print('Epoch: {}, iter: {}, loss: {}, acc: {}'.format(epoch, epoch * len(self.trainloader) + i, self.train_loss_metric.avg, self.train_acc_metric.avg))
            
            tqdm.write('Epoch: {}, iter: {}, loss: {}, acc: {}'.format(epoch, epoch * len(self.trainloader) + i, self.train_loss_metric.avg, self.train_acc_metric.avg))

    # uses the train_one_epoch repeatedly for findihing the training
    def train(self, start_epoch=0):
        #if self.cfg['train']['pretrained'] == "True":
        #    epoch = 

        for epoch in range(start_epoch, self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            epoch_acc = self.validate(epoch)
            if epoch_acc > self.best_val_acc:
                self.best_val_acc = epoch_acc
                if not os.path.exists(self.cfg['output_dir']):
                    os.makedirs(self.cfg['output_dir'])

                saved_name = os.path.join(self.cfg['output_dir'], '{}_{}_best.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
                print('best saved as', saved_name)
                #saves all the necessary information
                state = {
                    'epoch': epoch,
                    'state_dict': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                
                torch.save(state, saved_name)
            self.save_model(epoch)
            print("validation accuracy: ", epoch_acc)
        print("Best validation accuracy: ", self.best_val_acc)
        test_acc = self.test_accuracy()
        print("Test accuracy: ", test_acc)
        APCER, BPCER, ACER = self.calculate_metrics()
        print("APCER: ", APCER)
        print("BPCER: ", BPCER)
        print("ACER: ", ACER)


    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        seed = randint(0, len(self.valloader)-1)
        with torch.no_grad():
            for i, (img, depth_map, label) in tqdm(enumerate(self.valloader), total=len(self.valloader)):
                img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
                net_depth_map = self.network(img)
                loss = self.criterion(net_depth_map, depth_map)

                preds, score = predict(net_depth_map)
                targets, _ = predict(depth_map)

                accuracy = calc_accuracy(preds, targets)

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)

                if i == seed:
                    add_visualization_to_tensorboard(self.cfg, epoch, img, preds, targets, score, self.writer)

            return self.val_acc_metric.avg

    # def test_accuracy(self):
    #     self.network.eval()

    #     with torch.no_grad():
    #         for i, (img, depth_map, label) in tqdm(enumerate(self.testloader), total=len(self.testloader)):
    #             img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
    #             net_depth_map = self.network(img)
    #             loss = self.criterion(net_depth_map, depth_map)

    #             preds, score = predict(net_depth_map)
    #             targets, _ = predict(depth_map)

    #             accuracy = calc_accuracy(preds, targets)

    #             # Update metrics
    #             self.test_loss_metric=loss.item()
    #             self.test_acc_metric=accuracy

    #             # add_visualization_to_tensorboard(self.cfg, -1, img, preds, targets, score, self.writer)  # epoch set to -1

    #         return self.test_acc_metric

    def test_accuracy(self):
        self.network.eval()
        self.test_loss_metric.reset(0)
        self.test_acc_metric.reset(0)
        with torch.no_grad():
            for i, (img, depth_map, label) in tqdm(enumerate(self.testloader), total=len(self.testloader)):
                img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
                net_depth_map = self.network(img)
                loss = self.criterion(net_depth_map, depth_map)

                preds, score = predict(net_depth_map)
                targets, _ = predict(depth_map)

                accuracy = calc_accuracy(preds, targets)

                # Update metrics using AvgMeter's update method
                self.test_loss_metric.update(loss.item())
                self.test_acc_metric.update(accuracy)

                # add_visualization_to_tensorboard(self.cfg, -1, img, preds, targets, score, self.writer)  # epoch set to -1

            # After the loop, you can retrieve the average values using AvgMeter's properties
            return self.test_acc_metric.avg




    def calculate_metrics(self):
        self.network.eval()

        with torch.no_grad():
            TP, FP, TN, FN = 0, 0, 0, 0

            for i, (img, depth_map, label) in tqdm(enumerate(self.testloader), total=len(self.testloader)):
                img, depth_map, label = img.to(self.device), depth_map.to(self.device), label.to(self.device)
                net_depth_map = self.network(img)

                preds, _ = predict(net_depth_map)
                targets, _ = predict(depth_map)

                true_positives, false_positives, true_negatives, false_negatives = confusion(preds, targets)

                TP += true_positives
                FP += false_positives
                TN += true_negatives
                FN += false_negatives

                loss = self.criterion(net_depth_map, depth_map)

                # add_visualization_to_tensorboard(self.cfg, -1, img, preds, targets, score, self.writer)  # epoch set to -1

            # Calculate APCER, BPCER, and ACER
            APCER = (FP / (TN + FP)) * 100
            BPCER = (FN / (FN + TP)) * 100
            ACER = (APCER + BPCER) / 2

            return APCER, BPCER, ACER