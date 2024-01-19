from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from model import SR_model
from dataset import LR_dataset
from trainer import Trainer
# from parsing import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])  
learning_rate = 1e-3
batch_size = 32
epochs = 5

# loader
train_dataset = LR_dataset(dir_path='./COLON/TRAIN_SLICES/HR2',SR_factor=6, mode='train', transform=transform)
valid_dataset = LR_dataset(dir_path='./COLON/TRAIN_SLICES/HR2',SR_factor=6, mode='valid', transform=transform)
test_dataset  = LR_dataset(dir_path='./COLON/TRAIN_SLICES/HR2',SR_factor=6, mode='test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= False, num_workers= 8)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle= False, num_workers= 8)
test_loader  = DataLoader(test_dataset , batch_size=batch_size, shuffle= False, num_workers= 8)


# model
model = SR_model().to(device)

# loss
criterion = nn.L1Loss()

# optim
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# trainer
t = Trainer(model, epochs,train_loader, criterion, optimizer, learning_rate, device, valid_loader)

# train
t.train()