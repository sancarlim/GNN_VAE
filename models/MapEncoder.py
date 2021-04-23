import torch
import torch.nn as nn
import torch.nn.functional as F



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