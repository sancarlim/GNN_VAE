import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50



class ResNet18(nn.Module):
    def __init__(self, hidden_dim, freeze=8):
        super(ResNet18, self).__init__()
        
        model_ft = resnet18(pretrained=True)
        self.base = nn.Sequential(*list(model_ft.children())[:-3])
        self.last_conv = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, hidden_dim),
        )
        #model_ft.fc = nn.Linear(model_ft.fc.in_features, hidden_dim)
        ct=0
        for child in self.base.children():
            ct+=1
            if ct < freeze:  #if 7 freeze 2 BasicBlocks , train last one 128 -> 256
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.base(x)
        x = self.last_conv(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, hidden_dim, freeze=7):
        super(ResNet50, self).__init__()
        
        model_ft = resnet50(pretrained=True)
        self.base= nn.Sequential(*list(model_ft.children())[:6])
        #moduels.append(torch.)
        #modules.append(torch.nn.AdaptiveAvgPool2d((1, 1))) 
        #modules.append(torch.nn.Flatten(start_dim=1))
        self.last_conv = torch.nn.Sequential(
                            #nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1,bias=False),
                            #nn.BatchNorm2d(256),
                            #nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(),
                            nn.Linear(512, hidden_dim),
        ) 
        #model_ft.fc = nn.Linear(model_ft.fc.in_features, hidden_dim)
        ct=0
        for child in self.base.children():
            ct+=1
            if ct < freeze: 
                for param in child.parameters():
                    param.requires_grad = False
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for m in self.last_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = self.last_conv(x)
        return x



class My_MapEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_size, output_size, kernels, strides):
        super(My_MapEncoder, self).__init__()
        
        input_shape = (input_channels, input_size, input_size)
        #output_convs_size = torch.ones(input_shape).unsqueeze(0)
        self.conv1 = nn.Conv2d(input_channels, hidden_channels[0], kernel_size=kernels[0], stride=strides[0], bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        #output_convs_size = self.bn1(self.conv1(output_convs_size))
        
        self.convs = nn.ModuleList()
        for i in range(1,len(hidden_channels)-1):
            self.convs.append(nn.Sequential(
                                            nn.Conv2d(hidden_channels[i-1], hidden_channels[i], kernel_size=kernels[i], stride=strides[i], bias=False),
                                            nn.BatchNorm2d(hidden_channels[i]),
                                            nn.LeakyReLU(0.2, inplace=True)
            ))
        self.convs.append(nn.Sequential(
                                        nn.Conv2d(hidden_channels[-2], hidden_channels[-1], kernel_size=kernels[-1], stride=strides[-1]),
                                        nn.BatchNorm2d(hidden_channels[-1]),
            ))   
            #output_convs_size=self.convs[i-1](output_convs_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(hidden_channels[-1], output_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        for m in self.convs:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        for conv in self.convs:
            x = conv(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        maps_enc = self.fc(x)
        return maps_enc