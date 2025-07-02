import os
CURRENT_DIR = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(CURRENT_DIR, 'attack_model'))

import time
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms

from target_model import ConvNet, ResNet, ViT
from attack_model import AdvGAN, PAGAN

GPU_ID = "1"
SETTING = "target"              # target, attack
MODEL_NAME = "convnet"        # convnet, resnet50, resnet101, resnet152, vit
DATA_NAME = "caltech101"             # mnist, cifar10, stl10, fashion_mnist, caltech101, caltech256, oxford_iiit_pet

ATTACK_NAME = "pagan"          # advgan, pagan

if DATA_NAME == "mnist":
    NUM_CLS, DIM_CHL, IMAGE_SIZE, PATCH_SIZE = 10, 1, 28, 4
if DATA_NAME == "fashion_mnist":
    NUM_CLS, DIM_CHL, IMAGE_SIZE, PATCH_SIZE = 10, 1, 28, 4
elif DATA_NAME == "cifar10":
    NUM_CLS, DIM_CHL, IMAGE_SIZE, PATCH_SIZE = 10, 3, 32, 8
elif DATA_NAME == "stl10":
    NUM_CLS, DIM_CHL, IMAGE_SIZE, PATCH_SIZE = 10, 3, 96, 8

NUM_EPOCH = 30
BATCH_SIZE = 128

TIMESTAMP = time.strftime('%m%d_%H%M')
DATA_PATH = os.path.join(CURRENT_DIR, "data", DATA_NAME)

if SETTING == "target":
    TXT_LOG_FILE = os.path.join(CURRENT_DIR, "log", SETTING, f"{SETTING}_{MODEL_NAME}_{DATA_NAME}_{TIMESTAMP}.txt")
    TFB_LOG_FILE = os.path.join(CURRENT_DIR, "log", SETTING, f"{SETTING}_{MODEL_NAME}_{DATA_NAME}_{TIMESTAMP}")
elif SETTING == "attack":
    TXT_LOG_FILE = os.path.join(CURRENT_DIR, "log", SETTING, f"{SETTING}_{ATTACK_NAME}_{MODEL_NAME}_{DATA_NAME}_{TIMESTAMP}.txt")
    TFB_LOG_FILE = os.path.join(CURRENT_DIR, "log", SETTING, f"{SETTING}_{ATTACK_NAME}_{MODEL_NAME}_{DATA_NAME}_{TIMESTAMP}")

TARGET_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "target", f"target_{MODEL_NAME}_{DATA_NAME}.h5")
ATTACK_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "attack", f"attack_{ATTACK_NAME}_{MODEL_NAME}_{DATA_NAME}.h5")

def set_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    return device


def setup_txt_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_data(data_path, logger):    
    if DATA_NAME == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
    
    if DATA_NAME == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform)
    
    if DATA_NAME == "caltech101":
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])
        full_dataset = datasets.Caltech101(root=DATA_PATH, download=True, transform=transform)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    if DATA_NAME == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)

    if DATA_NAME == "stl10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.STL10(root=DATA_PATH, split='train', download=True, transform=transform)
        test_dataset = datasets.STL10(root=DATA_PATH, split='test', download=True, transform=transform)

    return train_dataset, test_dataset


def create_dataloader(train_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader


def train(device, logger, writer, model, save_model_path, num_epochs, train_dataloader, optimizer, loss_func):
    logger.info("Start Training")

    for epoch in range(num_epochs):
        train_loss = []
        train_corrects = 0
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.float().to(device)
            b_y = b_y.to(device)

            logit = model(b_x.float())
            output = torch.softmax(logit, dim=1)
            loss = loss_func(logit, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            train_loss.append(loss.item())
            train_corrects += pred.eq(b_y.view_as(pred)).float().sum().item()

        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_accuracy = 100.0 * train_corrects / len(train_dataloader.dataset)
        logger.info('[Epoch {:2d}/{}] Train - loss: {:.6f} acc: {:3.4f}%' .format(epoch, num_epochs, avg_train_loss, avg_train_accuracy))
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", avg_train_accuracy, epoch)

    torch.save(model.state_dict(), save_model_path)
    logger.info("Model Saved")


def eval(device, logger, model, test_loader):
    model.eval()
    with torch.no_grad():
        corrects = 0
        avg_loss = 0
        loss_func = nn.CrossEntropyLoss()
        for _, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            logit = model(b_x.float())
            loss = loss_func(logit, b_y)
            output = torch.softmax(logit, dim=1)
            pred = output.argmax(dim=1)
            corrects += pred.eq(b_y.view_as(pred)).float().sum().item()
            avg_loss += loss.item()
    avg_loss /= len(test_loader)
    accuracy = 100.0 * corrects / len(test_loader.dataset)
    logger.info(f"Number of corrects: {corrects}")
    logger.info("Test - loss: {:.6f} acc: {:3.4f}%".format(avg_loss, accuracy))


def target(device, logger, writer, target_model_path, num_epoch, num_cls, train_dataloader, test_dataloader):
    if MODEL_NAME == "resnet50":
        model = ResNet.ResNet50(num_classes=num_cls, channels=DIM_CHL).to(device)

    elif MODEL_NAME == "resnet101":
        model = ResNet.ResNet101(num_classes=num_cls, channels=DIM_CHL).to(device)
    
    elif MODEL_NAME == "resnet152":
        model = ResNet.ResNet152(num_classes=num_cls, channels=DIM_CHL).to(device)

    elif MODEL_NAME == "convnet":
        model = ConvNet.ConvNet4(num_classes=num_cls, channels=DIM_CHL).to(device)
    
    elif MODEL_NAME == "vit":
        model = ViT.ViT(
            image_size=128, patch_size=PATCH_SIZE, num_classes=num_cls, 
            dim=512, depth=6, heads=8, mlp_dim=1024,
            channels=DIM_CHL, dim_head=64, dropout=0.1, emb_dropout=0.1
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_func = nn.CrossEntropyLoss()
    train(device, logger, writer, model, target_model_path, num_epoch, train_dataloader, optimizer, loss_func)
    eval(device, logger, model, test_dataloader)


def attack(logger, writer, target_model_path, num_epoch, num_cls, train_dataloader, test_dataloader):
    logger.info("Start Attacking")
    if MODEL_NAME == "resnet50":
        targeted_model = ResNet.ResNet50(num_classes=num_cls, channels=DIM_CHL).to(device)

    elif MODEL_NAME == "resnet101":
        targeted_model = ResNet.ResNet101(num_classes=num_cls, channels=DIM_CHL).to(device)
    
    elif MODEL_NAME == "resnet152":
        targeted_model = ResNet.ResNet152(num_classes=num_cls, channels=DIM_CHL).to(device)

    elif MODEL_NAME == "convnet":
        targeted_model = ConvNet.ConvNet4(num_classes=num_cls, channels=DIM_CHL).to(device)
    
    elif MODEL_NAME == "vit":
        targeted_model = ViT.ViT(
            image_size=128, patch_size=PATCH_SIZE, num_classes=num_cls, 
            dim=512, depth=6, heads=8, mlp_dim=1024,
            channels=DIM_CHL, dim_head=64, dropout=0.1, emb_dropout=0.1
        ).to(device)

    targeted_model.load_state_dict(torch.load(target_model_path, weights_only=True))
    targeted_model.eval()
    logger.info("Target Model Loaded")

    if ATTACK_NAME == "advgan":
        attack_model = AdvGAN.AdvGAN(device, targeted_model, num_cls, DIM_CHL, IMAGE_SIZE)
    elif ATTACK_NAME == "pagan":
        attack_model = PAGAN.PAGAN(device, targeted_model, num_cls, DIM_CHL, IMAGE_SIZE)

    logger.info("Start Training")
    attack_model.train(ATTACK_MODEL_PATH, train_dataloader, num_epoch, logger)


if __name__ == "__main__":
    device = set_device(GPU_ID)
    logger = setup_txt_logger(SETTING, TXT_LOG_FILE)
    writer = SummaryWriter(TFB_LOG_FILE)

    if SETTING == "target":
        train_dataset, test_dataset = load_data(DATA_NAME, logger)
        train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset)
        target(device, logger, writer, TARGET_MODEL_PATH, NUM_EPOCH, NUM_CLS, train_dataloader, test_dataloader)
        
    elif SETTING == "attack":
        train_dataset, test_dataset = load_data(DATA_NAME, logger)
        train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset)
        attack(logger, writer, TARGET_MODEL_PATH, NUM_EPOCH, NUM_CLS, train_dataloader, test_dataloader)