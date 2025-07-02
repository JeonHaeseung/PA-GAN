import os
CURRENT_DIR = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(CURRENT_DIR, 'attack_model'))

import time
import logging
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from target_model import ConvNet, ResNet, ViT
from attack_model import AdvGAN, PAGAN

import GAN
from GAN import AdvDiscriminator, AdvGenerator

GPU_ID = "1"
MODEL_NAME = "resnet50"          # convnet, resnet50, resnet101, resnet152, vit
DATA_NAME = "stl10"           # mnist, cifar10, stl10, caltech101, caltech256, oxford-IIIT-pet
ATTACK_NAME = "pagan"           # advgan, pagan
MAX = 0.3

if DATA_NAME == "mnist":
    NUM_CLS, DIM_CHL, IMAGE_SIZE, PATCH_SIZE = 10, 1, 28, 4
elif DATA_NAME == "cifar10":
    NUM_CLS, DIM_CHL, IMAGE_SIZE, PATCH_SIZE = 10, 3, 32, 8
elif DATA_NAME == "stl10":
    NUM_CLS, DIM_CHL, IMAGE_SIZE, PATCH_SIZE = 10, 3, 96, 8

BATCH_SIZE = 128
GEN_INPUT_NC = DIM_CHL

TIMESTAMP = time.strftime('%m%d_%H%M')
DATA_PATH = os.path.join(CURRENT_DIR, "data", DATA_NAME)
TXT_LOG_FILE = os.path.join(CURRENT_DIR, "log", "eval", f"eval_{ATTACK_NAME}_{MODEL_NAME}_{DATA_NAME}_{MAX}_{TIMESTAMP}.txt")

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


def load_target(target_model_path, num_cls, device):
    # load the pretrained model
    if MODEL_NAME == "resnet50":
        target_model = ResNet.ResNet50(num_classes=num_cls, channels=DIM_CHL).to(device)

    elif MODEL_NAME == "resnet101":
        target_model = ResNet.ResNet101(num_classes=num_cls, channels=DIM_CHL).to(device)
    
    elif MODEL_NAME == "resnet152":
        target_model = ResNet.ResNet152(num_classes=num_cls, channels=DIM_CHL).to(device)

    elif MODEL_NAME == "convnet":
        target_model = ConvNet.ConvNet4(num_classes=num_cls, channels=DIM_CHL).to(device)
    
    elif MODEL_NAME == "vit":
        target_model = ViT.ViT(
            image_size=128, patch_size=PATCH_SIZE, num_classes=num_cls, 
            dim=512, depth=6, heads=8, mlp_dim=1024,
            channels=DIM_CHL, dim_head=64, dropout=0.1, emb_dropout=0.1
        ).to(device)

    target_model.load_state_dict(torch.load(target_model_path))
    target_model.eval()

    return target_model


def load_attack(attack_model_path):
    # load the generator of adversarial examples
    pretrained_G = AdvGenerator(GEN_INPUT_NC, DIM_CHL, IMAGE_SIZE).to(device)
    pretrained_G.load_state_dict(torch.load(attack_model_path))
    pretrained_G.eval()
    
    return pretrained_G


def load_dataset():
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

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, test_dataloader


def test(logger, target_model, attack_model, train_dataloader, test_dataloader):
    # test adversarial examples in MNIST training dataset
    train_total_samples = len(train_dataloader.dataset)
    ori_num_correct = 0
    adv_num_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img),1)
        ori_num_correct += torch.sum(pred_lab==test_label,0)

        perturbation = attack_model(test_img)
        perturbation = torch.clamp(perturbation, -MAX, MAX)

        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        adv_num_correct += torch.sum(pred_lab==test_label,0)

    logger.info(f'ori_num_correct: {ori_num_correct.item()}')
    logger.info(f'accuracy of ori imgs in training set: {(ori_num_correct.item()/train_total_samples)}')
    logger.info(f'adv_num_correct: {adv_num_correct.item()}')
    logger.info(f'accuracy of adv imgs in training set: {(adv_num_correct.item()/train_total_samples)}')


    # test adversarial examples in MNIST testing dataset
    test_total_samples = len(test_dataloader.dataset)
    ori_num_correct = 0
    adv_num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img),1)
        ori_num_correct += torch.sum(pred_lab==test_label,0)

        perturbation = attack_model(test_img)
        perturbation = torch.clamp(perturbation, -MAX, MAX)
        
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        adv_num_correct += torch.sum(pred_lab==test_label,0)

    logger.info(f'ori_num_correct: {ori_num_correct.item()}')
    logger.info(f'accuracy of ori imgs in testing set: {(ori_num_correct.item()/test_total_samples)}')
    logger.info(f'adv_num_correct: {adv_num_correct.item()}')
    logger.info(f'accuracy of adv imgs in testing set: {(adv_num_correct.item()/test_total_samples)}')


if __name__ == "__main__":
    device = set_device(GPU_ID)
    logger = setup_txt_logger("eval", TXT_LOG_FILE)
    target_model = load_target(TARGET_MODEL_PATH, NUM_CLS, device)
    attack_model = load_attack(ATTACK_MODEL_PATH)
    train_dataloader, test_dataloader = load_dataset()
    test(logger, target_model, attack_model, train_dataloader, test_dataloader)