import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, mix_size, low_level_inplanes, num_classes, BatchNorm):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.last_conv = nn.Sequential(nn.Conv2d(48+mix_size, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self.theta = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                    BatchNorm(48))
        self.psi = nn.Conv2d(48, 1, 1, bias=True)
        self.W = nn.Sequential(nn.Conv2d(48, 48, 1, 1, 0, bias=False),
                               BatchNorm(48))
        self._init_weight()

    def attention_connection(self, low_level_feat, x):
        theta_x = self.theta(x)
        g = F.interpolate(low_level_feat, x.shape[2:], mode='bilinear', align_corners=True)
        f = F.softplus(theta_x + g) # soft relu
        att = torch.sigmoid(self.psi(f))
        att = F.interpolate(att, low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        y = att.expand_as(low_level_feat) * low_level_feat
        W_y = self.W(y)
        return W_y, att

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat, att = self.attention_connection(low_level_feat, x)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
