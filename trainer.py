from tqdm import tqdm
import torch

class Trainer():
    def __init__(self, model, epochs, train_loader, criterion, optimizer, lr,device,valid_loader=None, test_loader=None):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.valid_loader = valid_loader
        self.lr = lr
        self.trainlosses = []
        self.validlosses = []

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            total_train_loss = 0
            total_valid_loss = 0
            for LR_input, HR_gt in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                LR_input = LR_input.to(self.device)
                HR_gt = HR_gt.to(self.device)

                output = self.model(LR_input)
                loss = self.criterion(HR_gt,output)
                total_train_loss+= loss.item()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch+1} | Train loss: {total_train_loss / len(self.train_loader):.4f} |')
            self.trainlosses.append(total_train_loss/len(self.train_loader))
            self.validlosses.append(total_valid_loss/len(self.valid_loader))
    
    def test(self):
        self.model.eval()
        total_test_loss = 0.0

        with torch.no_grad():
            for LR_input, HR_gt in tqdm(self.test_loader):
                LR_input = LR_input.to(self.device)
                HR_gt = HR_gt.to(self.device)
                outputs = self.model(LR_input)
                loss = self.criterion(outputs,HR_gt)
                total_test_loss += loss.item()

