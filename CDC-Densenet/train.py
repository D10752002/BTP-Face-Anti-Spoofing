import os
import torch
from torchvision import transforms, models
from tensorboardX import SummaryWriter
from datasets.FASDataset import FASDataset
from utils.transform import RandomGammaCorrection
from utils.utils import read_cfg, get_optimizer, get_device, build_network
from trainer.FASTrainer import FASTrainer
from models.loss import DepthLoss
from torch.optim.lr_scheduler import StepLR

def main():
    cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")
    
    device = get_device(cfg)
    
    network = build_network(cfg)
    
    optimizer = get_optimizer(cfg, network)
    
    lr_scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
    
    criterion = DepthLoss(device=device)
    
    writer = SummaryWriter(cfg['log_dir'])

    dump_input = torch.randn((1, 3, cfg['model']['input_size'][0], cfg['model']['input_size'][1]))
    

    
    train_transform = transforms.Compose([
        RandomGammaCorrection(max_gamma=cfg['dataset']['augmentation']['gamma_correction'][1],
                                min_gamma=cfg['dataset']['augmentation']['gamma_correction'][0]),
        transforms.RandomResizedCrop(cfg['model']['input_size'][0]),
        # transforms.ColorJitter(
        #     brightness=cfg['dataset']['augmentation']['brightness'],
        #     contrast=cfg['dataset']['augmentation']['contrast'],
        #     saturation=cfg['dataset']['augmentation']['saturation'],
        #     hue=cfg['dataset']['augmentation']['hue']
        # ),
        transforms.RandomRotation(cfg['dataset']['augmentation']['rotation_range']),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(cfg['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(cfg['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(cfg['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
    ])
    
    trainset = FASDataset(
        root_dir=cfg['dataset']['root'],
        csv_file=cfg['dataset']['train_set'],
        depth_map_size=cfg['model']['depth_map_size'],
        transform=train_transform,
        smoothing=cfg['train']['smoothing']
    )
    
    valset = FASDataset(
        root_dir=cfg['dataset']['root'],
        csv_file=cfg['dataset']['val_set'],
        depth_map_size=cfg['model']['depth_map_size'],
        transform=val_transform,
        smoothing=cfg['train']['smoothing']
    )

    testset = FASDataset(
        root_dir=cfg['dataset']['root'],
        csv_file=cfg['dataset']['test_set'],
        depth_map_size=cfg['model']['depth_map_size'],
        transform=test_transform,
        smoothing=cfg['train']['smoothing']
    )
    
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=8
    )

    valloader = torch.utils.data.DataLoader(
        dataset=valset,
        batch_size=cfg['val']['batch_size'],
        shuffle=True,
        num_workers=8
    )
    
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=cfg['test']['batch_size'],
        shuffle=True,
        num_workers=8
    )
    
    trainer = FASTrainer(
        cfg=cfg, 
        network=network,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        device=device,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        writer=writer
    )
    
    trainer.load_model(4)
    # Calculate test metrics
    # APCER, BPCER, ACER = trainer.calculate_metrics()

    # # Print the metrics
    # print("APCER: ", APCER)
    # print("BPCER: ", BPCER)
    # print("ACER: ", ACER)

    test_accuracy= trainer.test_accuracy()
    print("Test accuracy: ", test_accuracy)



    
    writer.close()
if __name__ == '__main__':
    main()
    #freeze_support()