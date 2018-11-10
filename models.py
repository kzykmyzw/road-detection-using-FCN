import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class enc(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(enc, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class dec(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(dec, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class encdec(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(encdec, self).__init__()

        self.enc1 = enc(in_channels, 64, normalize=False)
        self.enc2 = enc(64, 128)
        self.enc3 = enc(128, 256)
        self.enc4 = enc(256, 512, dropout=0.5)
        self.enc5 = enc(512, 512, dropout=0.5)
        self.enc6 = enc(512, 512, dropout=0.5)
        self.enc7 = enc(512, 512, dropout=0.5)
        self.enc8 = enc(512, 512, normalize=False, dropout=0.5)

        self.dec1 = dec(512, 512, dropout=0.5)
        self.dec2 = dec(1024, 512, dropout=0.5)
        self.dec3 = dec(1024, 512, dropout=0.5)
        self.dec4 = dec(1024, 512, dropout=0.5)
        self.dec5 = dec(1024, 256)
        self.dec6 = dec(512, 128)
        self.dec7 = dec(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # encoder-decoder with skip connection
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        d1 = self.dec1(e8, e7)
        d2 = self.dec2(d1, e6)
        d3 = self.dec3(d2, e5)
        d4 = self.dec4(d3, e4)
        d5 = self.dec5(d4, e3)
        d6 = self.dec6(d5, e2)
        d7 = self.dec7(d6, e1)

        return self.final(d7)
