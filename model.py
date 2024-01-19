# Model
import torch
import torch.nn as nn

def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers) # *으로 list unpacking 

            return cbr

def CBR2d_(in_channels, out_channels, kernel_size=3, stride=1, padding=(1,2), bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers) # *으로 list unpacking 

            return cbr

def CBR2d__(in_channels, out_channels, kernel_size=3, stride=1, padding=(1,0), bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers) # *으로 list unpacking 

            return cbr

class SR_model(nn.Module):
    def __init__(self):
        super(SR_model, self).__init__()

        # Contracting path
        self.enc1_1 = CBR2d_(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d_(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128) 
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256) 
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512) 
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512) 
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottle_neck1 = CBR2d(in_channels=512, out_channels=1024) 

        # Expansive path
        self.bottle_neck2 = CBR2d(in_channels=1024, out_channels=1024)

        self.unpool1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, 
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_1 = CBR2d(in_channels=2 * 512, out_channels=512) 
        self.dec1_2 = CBR2d(in_channels=512, out_channels=512) 

        self.unpool2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, 
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_1 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec2_2 = CBR2d(in_channels=256, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_1 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec3_2 = CBR2d(in_channels=128, out_channels=128)

        self.unpool4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_1 = CBR2d__(in_channels=2 * 64, out_channels=64)
        self.dec4_2 = CBR2d__(in_channels=64, out_channels=1)

    def forward(self, x):
        # input = (B,1,512,60)
        enc1_1 = self.enc1_1(x)
        # print(enc1_1.shape) = (B,64,512,62)
        enc1_2 = self.enc1_2(enc1_1)
        # print(enc1_2.shape) = (B,64,512,64)
        pool1 = self.pool1(enc1_2) 
        # (B,64,256,32)

        enc2_1 = self.enc2_1(pool1) # (B,128,256,32)
        enc2_2 = self.enc2_2(enc2_1) # (B,128,256,32)
        pool2 = self.pool2(enc2_2) # (B,128,128,16)

        enc3_1 = self.enc3_1(pool2) # (B,256,128,16)
        enc3_2 = self.enc3_2(enc3_1) # (B,256,128,16)
        pool3 = self.pool3(enc3_2) # (B,256,64,8)

        enc4_1 = self.enc4_1(pool3) # (B,512,64,8)
        enc4_2 = self.enc4_2(enc4_1) # (B,512,64,8)
        pool4 = self.pool2(enc4_2) # (B,512,64,8)

        bottle_neck1 = self.bottle_neck1(pool4) # (B,1024,64,8)
        bottle_neck2 = self.bottle_neck2(bottle_neck1) # (B,1024,64,8)
        
        unpool1 = self.unpool1(bottle_neck2) # (B,512,128,16)
        cat1 = torch.cat((unpool1, enc4_2), dim=1) # (B, 1024, 128, 16)
        dec1_1 = self.dec1_1(cat1) # (B, 512, 128, 16)
        dec1_2 = self.dec1_2(dec1_1) # (B, 512, 128, 16)

        unpool2 = self.unpool2(dec1_2) # (B, 256, 256, 32)

        cat2 = torch.cat((unpool2, enc3_2), dim=1) # (B, 512, 256, 32)
        dec2_1 = self.dec2_1(cat2) # (B, 256, 256, 32)
        dec2_2 = self.dec2_2(dec2_1) # (B, 256, 256, 32)

        unpool3 = self.unpool3(dec2_2) # (B, 128, 512, 64)
        cat3 = torch.cat((unpool3, enc2_2), dim=1) # (B, 256, 512, 64)
        dec3_1 = self.dec3_1(cat3) # (B, 128, 512, 64)
        dec3_2 = self.dec3_2(dec3_1) # (B, 128, 512, 64)

        unpool4 = self.unpool4(dec3_2) # (B, 64, 512, 64)
        cat4 = torch.cat((unpool4, enc1_2), dim=1) # (B, 128, 512, 64)
        dec4_1 = self.dec4_1(cat4) # (B, 64, 512, 62)
        output = self.dec4_2(dec4_1) # (B, 1, 512, 60)

        return output # (B, 1, 512, 60)