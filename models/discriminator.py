import torch 
import torch.nn as nn
from .generator import Generator

from torch.nn.init import calculate_gain, kaiming_normal_


class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
        
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)
    

class DownsampleLayer(nn.Module):
    def __init__(self, in_dim, k_filters, k_width, k_heigth, padding='same', ksize_down=2, stride_down=2, bias=False):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=ksize_down, stride=stride_down, ceil_mode=False, count_include_pad=False),
            
            nn.Conv2d(in_dim, k_filters, kernel_size=(k_width,k_heigth), padding=padding, bias=bias),
            nn.LeakyReLU(.2),
            
            nn.Conv2d(k_filters, k_filters, kernel_size=(k_width,k_heigth), padding=padding, bias=bias),
            nn.LeakyReLU(.2),
        )
                
    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, k_width=3, k_heigth=3, k_filters=32, padding='same', scale_factor=2, ksize_down=2, stride_down=2):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_width = k_width
        self.k_height = k_heigth
        self.k_filters = k_filters
        self.scale_factor = scale_factor
        self.ksize_down = ksize_down
        self.stride_down = stride_down
    
               
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, self.k_filters, kernel_size=(1,1), padding='valid', bias=False),
            
            nn.Conv2d(self.k_filters, self.k_filters, kernel_size=(self.k_width,self.k_height), padding=padding, bias=False),
            nn.LeakyReLU(.2),
            
            nn.Conv2d(self.k_filters, self.k_filters, kernel_size=(self.k_width,self.k_height), padding=padding, bias=False),
            nn.LeakyReLU(.2),
            
            DownsampleLayer(self.k_filters, self.k_filters*2, self.k_width, self.k_height, padding=padding, ksize_down=self.ksize_down, bias=False),
            DownsampleLayer(self.k_filters*2, self.k_filters*(2**2), self.k_width, self.k_height,  padding=padding, ksize_down=self.ksize_down, bias=False),
            DownsampleLayer(self.k_filters*(2**2), self.k_filters*(2**3), self.k_width, self.k_height,  padding=padding, ksize_down=self.ksize_down, bias=False),
            DownsampleLayer(self.k_filters*(2**3), self.k_filters*(2**3), self.k_width, self.k_height,  padding=padding, ksize_down=self.ksize_down, bias=False),
            DownsampleLayer(self.k_filters*(2**3), self.k_filters*(2**3), self.k_width, self.k_height, padding=padding, ksize_down=self.ksize_down, bias=False),
            DownsampleLayer(self.k_filters*(2**3), self.k_filters*(2**3), self.k_width, self.k_height, padding=padding, ksize_down=self.ksize_down, bias=False),
            
            nn.Flatten(),
            nn.Linear(self.k_filters*(2**3)*2*16, self.out_channels),
        )
        
        
    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d)):
                kaiming_normal_(m.weight, a=calculate_gain('conv2d'))
        
        
    def forward(self, x):
        output = self.model(x)
        return output




#######################################################################################
# Just a test function to verify the right behavior of the generator and discriminator
#######################################################################################
def test_disc(TEST=False):
    if TEST:
        noise = torch.randn(8,256)
        
        gen = Generator()
        gen.initialize_weights()
        
        disc = Discriminator()
        disc.initialize_weights()

        fake = gen(noise)
        print(fake[0:2])
        result = disc(fake)
        print(fake.size(), result.size())
        print(result)
    
test_disc(False)