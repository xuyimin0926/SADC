import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import time
import kornia

def conv(X, W, s):
    x1_use = X[:,:,s,:,:]
    x1_out = torch.einsum('ncskj,dckj->nds',x1_use,W)
    return x1_out

class DynConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation = 1, bias = True, groups = 1, norm = 'in', nonlinear = 'relu'):
        super(DynConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1)*(kernel_size - 1))//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias = bias, dilation = dilation)
        self.augment_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias = bias, dilation = dilation)
        self.light_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=in_channels, bias = bias, dilation = dilation)
        self.norm = norm
        self.nonlinear = nonlinear
        self.dilation = dilation
        self.groups = in_channels

        if norm == 'bn':
            self.normalization = nn.BatchNorm2d(out_channels)
            self.augment_normalization = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.normalization = nn.InstanceNorm2d(out_channels, affine = False)
            self.augment_normalization = nn.InstanceNorm2d(out_channels, affine = False)
        else:
            self.normalization = None
            self.augment_normalization = None
          
        if nonlinear == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif nonlinear == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
            self.activation = nn.PReLU()
        else:
            self.activation = None

    def forward(self, x, mask, mask_d, mask_e):

        if self.training:
            out_shadow = self.conv2d(self.reflection_pad(x))
            if self.normalization is not None:
                out_shadow = self.normalization(out_shadow)
            if self.activation is not None:
                out_shadow = self.activation(out_shadow)

            out_shadow = self.augment_conv2d(self.reflection_pad(out_shadow))
            if self.normalization is not None:
                out_shadow = self.normalization(out_shadow)
            if self.activation is not None:
                out_shadow = self.activation(out_shadow)

            # Unshadow region color mapping
            out_unshadow = self.light_conv2d(self.reflection_pad(x))
            if self.normalization is not None:
                out_unshadow = self.normalization(out_unshadow)
            if self.activation is not None:
                out_unshadow = self.activation(out_unshadow)

            for i, out_s in enumerate(out_shadow):
                if i == 0:
                    if mask[i].sum() <= 100 or mask[i].sum() > 0.99 * mask[i].numel():
                        out_shadow_mean = out_shadow[i].mean(dim=(1, 2)).unsqueeze(0)
                        out_unshadow_mean = out_unshadow[i].mean(dim=(1, 2)).unsqueeze(0)
                        continue
                    out_shadow_mean = out_shadow[i][(mask[i] - mask_e[i]).expand_as(x[i]) == 1].reshape(
                        (x.shape[1], -1)).mean(-1).unsqueeze(0)
                    out_unshadow_mean = out_unshadow[i][(mask_d[i] - mask[i]).expand_as(x[i]) == 1].reshape(
                        (x.shape[1], -1)).mean(-1).unsqueeze(0)
                else:
                    if mask[i].sum() <= 100 or mask[i].sum() > 0.99 * mask[i].numel():
                        out_shadow_mean = torch.cat((out_shadow_mean, out_shadow[i].mean(dim=(1, 2)).unsqueeze(0)),
                                                    dim=0)
                        out_unshadow_mean = torch.cat(
                            (out_unshadow_mean, out_unshadow[i].mean(dim=(1, 2)).unsqueeze(0)), dim=0)
                        continue
                    out_shadow_mean = torch.cat((out_shadow_mean,
                                                 out_shadow[i][(mask[i] - mask_e[i]).expand_as(x[i]) == 1].reshape(
                                                     (x.shape[1], -1)).mean(dim=-1).unsqueeze(0)), dim=0)
                    out_unshadow_mean = torch.cat((out_unshadow_mean,
                                                   out_unshadow[i][(mask_d[i] - mask[i]).expand_as(x[i]) == 1].reshape(
                                                       (x.shape[1], -1)).mean(-1).unsqueeze(0)), dim=0)
            return mask * (out_shadow + x) + (1 - mask) * (out_unshadow + x), out_shadow_mean, out_unshadow_mean

        else:
            #T1 = time.time()
            shape = x.shape
            Bx,Cx,Hx,Wx = x.shape
            dilation = self.dilation

            mask_d1 = kornia.morphology.dilation(mask, torch.ones(self.dilation * 2 + 1, self.dilation * 2 + 1).cuda())

            md = torch.flatten(mask_d1).bool()
            sd = torch.flatten(mask).bool()

            x_ori =x.clone()

            w_conv_1 = self.conv2d.weight
            FN, C, ksize1, ksize, = w_conv_1.shape
            x1 = self.reflection_pad(x_ori)
            x_k = torch.nn.functional.unfold(x1, ksize, dilation=dilation, stride=1) #N*(Ckk)*(hw)
            x_k_k = x_k.reshape(x_k.shape[0],Cx,ksize,ksize,x_k.shape[2]).permute(0,1,4,2,3) #N*C*(hw)*k*k
            out_shadow_conv = conv(x_k_k,w_conv_1,md)+self.conv2d.bias.unsqueeze(0).unsqueeze(-1) #N*C*num(mask)

            if self.normalization is not None:
                out_shadow_conv = self.normalization(out_shadow_conv)
            if self.activation is not None:
                out_shadow_conv = self.activation(out_shadow_conv)


            x_ori = x_ori.reshape(x_ori.shape[0],x_ori.shape[1],-1)
            x_ori[:, :, md] = out_shadow_conv
            x_ori = x_ori.reshape(shape)

            x1 = self.reflection_pad(x_ori)
            x_k = torch.nn.functional.unfold(x1, ksize, dilation=dilation, stride=1)
            x_k_k = x_k.reshape(x_k.shape[0], Cx, ksize, ksize, x_k.shape[2]).permute(0, 1, 4, 2, 3)
            w_conv_2 = self.augment_conv2d.weight
            out_shadow_conv = conv(x_k_k, w_conv_2, sd) + self.augment_conv2d.bias.unsqueeze(0).unsqueeze(-1)

            if self.normalization is not None:
                out_shadow_conv = self.normalization(out_shadow_conv)
            if self.activation is not None:
                out_shadow_conv = self.activation(out_shadow_conv)

            ns = ~sd
            x1 = self.reflection_pad(x)
            x_k = torch.nn.functional.unfold(x1, ksize, dilation=dilation, stride=1)  # N*(Ckk)*(hw)
            x_k_k = x_k.reshape(x_k.shape[0], Cx, ksize, ksize, x_k.shape[2]).permute(0, 1, 4, 2, 3)  # N*C*(hw)*k*k
            x_k_k = x_k_k.reshape(x_k_k.shape[0], self.groups, -1, x_k_k.shape[2], x_k_k.shape[3],x_k_k.shape[4])  # N*g*(C/g)*(hw)*k*k
            w_conv_3 = self.light_conv2d.weight
            d, c_g, k, j = w_conv_3.shape
            w_conv_3_1 = w_conv_3.reshape(self.groups, -1, c_g, k, j)  # g*(d/g)*c_g*k*k
            x1_use = x_k_k[:, :, :, ns, :, :]
            x1_out = torch.einsum('ngcskj,gdckj->ngds', x1_use, w_conv_3_1)
            x1_out_1 = x1_out.reshape(x1_out.shape[0], d, x1_out.shape[3])
            out_unshadow_conv = x1_out_1 + self.light_conv2d.bias.unsqueeze(0).unsqueeze(-1)

            if self.normalization is not None:
                out_unshadow_conv = self.normalization(out_unshadow_conv)
            if self.activation is not None:
                out_unshadow_conv = self.activation(out_unshadow_conv)

            x_ori = x_ori.reshape(x_ori.shape[0],x_ori.shape[1],-1)
            x_ori[:, :, sd] = out_shadow_conv
            x_ori[:, :, ns] = out_unshadow_conv
            x_ori = x_ori.reshape(shape)

            out_shadow =  x_ori
            out_unshadow =  x_ori

            x = x_ori+x
            #T2 = time.time()
            #print('runtime:%s ms' % ((T2 - T1) * 1000))

            return x, out_shadow, out_unshadow

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation = 1, bias = True, groups = 1, norm = 'in', nonlinear = 'relu'):
        super(ConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1)*(kernel_size - 1))//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups = groups, bias = bias, dilation = dilation)
        self.norm = norm
        self.nonlinear = nonlinear
        
        if norm == 'bn':
            self.normalization = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.normalization = nn.InstanceNorm2d(out_channels, affine = False)
        else:
            self.normalization = None
          
        if nonlinear == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif nonlinear == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
            self.activation = nn.PReLU()
        else:
            self.activation = None
          
    def forward(self, x): 
        out = self.conv2d(self.reflection_pad(x))
        if self.normalization is not None:
            out = self.normalization(out)
        if self.activation is not None:
            out = self.activation(out)
        
        return out

class Self_Attention(nn.Module):
    def __init__(self, channels, k, nonlinear = 'relu'):
      super(Self_Attention, self).__init__()
      self.channels = channels
      self.k = k
      self.nonlinear = nonlinear
      
      self.linear1 = nn.Linear(channels, channels//k)
      self.linear2 = nn.Linear(channels//k, channels)
      self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
      
      if nonlinear == 'relu':
          self.activation = nn.ReLU(inplace = True)
      elif nonlinear == 'leakyrelu':
          self.activation = nn.LeakyReLU(0.2)
      elif nonlinear == 'PReLU':
          self.activation = nn.PReLU()
      else:
          raise ValueError
      
    def attention(self, x):
      N, C, H, W = x.size()
      out = torch.flatten(self.global_pooling(x), 1)
      out = self.activation(self.linear1(out))
      out = torch.sigmoid(self.linear2(out)).view(N, C, 1, 1)
      
      return out.mul(x)
      
    def forward(self, x):
      return self.attention(x)
      
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 4, interpolation_type = 'bilinear'):
      super(SPP, self).__init__()
      self.conv = nn.ModuleList()
      self.num_layers = num_layers
      self.interpolation_type = interpolation_type
      
      for _ in range(self.num_layers):
        self.conv.append(ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, dilation = 1, nonlinear = 'leakyrelu', norm = None))
      
      self.fusion = ConvLayer((in_channels*(self.num_layers+1)), out_channels, kernel_size = 3, stride = 1, norm = 'False', nonlinear = 'leakyrelu')
    
    def forward(self, x):
      
      N, C, H, W = x.size()
      out = []
      
      for level in range(self.num_layers):
        out.append(F.interpolate(self.conv[level](F.avg_pool2d(x, kernel_size = 2*2**(level+1), stride = 2*2**(level+1), padding = 2*2**(level+1)%2)), size = (H, W), mode = self.interpolation_type))      
      
      out.append(x)
      
      return self.fusion(torch.cat(out, dim = 1))

class Aggreation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
      super(Aggreation, self).__init__()
      self.attention = Self_Attention(in_channels, k = 8, nonlinear = 'relu')
      self.conv = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, dilation = 1, nonlinear = 'leakyrelu', norm = None)
      
    def forward(self, x):
      
      return self.conv(self.attention(x))
      

class Backbone(nn.Module):
    def __init__(self, backbones = 'vgg16'):
      super(Backbone, self).__init__()
      
      if backbones == 'vgg16':
        modules = (models.vgg16(pretrained = True).features[:-1])
    
        self.block1 = modules[0:4]
        self.block2 = modules[4:9]
        self.block3 = modules[9:16]
        self.block4 = modules[16:23]
        self.block5 = modules[23:]
      
        for param in self.parameters():
            param.requires_grad = False
      
      else:
        raise ValueError
        
    def forward(self, x):
        N, C, H, W = x.size()
        
        out = [x]
        
        out.append(self.block1(out[-1]))
        out.append(self.block2(out[-1]))
        out.append(self.block3(out[-1]))
        out.append(self.block4(out[-1]))
        out.append(self.block5(out[-1]))
        
        return torch.cat([(F.interpolate(item, size = (H, W), mode = 'bicubic') if sum(item.size()[2:]) != sum(x.size()[2:]) else item) for item in out], dim = 1).detach()

class ShadowRemoval(nn.Module):
    def __init__(self, channels = 64):
        super(ShadowRemoval, self).__init__()
        
        self.backbone = Backbone()
        self.fusion = ConvLayer(in_channels = 1475, out_channels = channels, kernel_size = 1, stride = 1, norm = None, nonlinear = 'leakyrelu')

        ##Stage0
        self.block0_1 = DynConvLayer(in_channels = channels, out_channels = channels, kernel_size = 1, stride = 1, norm = None, nonlinear = 'leakyrelu')
        self.block0_2 = DynConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, norm = None, nonlinear = 'leakyrelu')

        self.aggreation0_rgb = Aggreation(in_channels = channels*2, out_channels = channels)
        self.aggreation0_mas = Aggreation(in_channels = channels*2, out_channels = channels)
        
        ##Stage1
        self.block1_1 = DynConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 2, norm = None, nonlinear = 'leakyrelu')  
        self.block1_2 = DynConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 4, norm = None, nonlinear = 'leakyrelu')

        self.aggreation1_rgb = Aggreation(in_channels = channels*3, out_channels = channels)
        self.aggreation1_mas = Aggreation(in_channels = channels*3, out_channels = channels)
        
        ##Stage2
        self.block2_1 = DynConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 8, norm = None, nonlinear = 'leakyrelu')
        self.block2_2 = DynConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 16, norm = None, nonlinear = 'leakyrelu')

        self.aggreation2_rgb = Aggreation(in_channels = channels*3, out_channels = channels)
        self.aggreation2_mas = Aggreation(in_channels = channels*3, out_channels = channels)
        
        ##Stage3
        self.block3_1 = DynConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 32, norm = None, nonlinear = 'leakyrelu')
        self.block3_2 = DynConvLayer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, dilation = 64, norm = None, nonlinear = 'leakyrelu')

        self.aggreation3_rgb = Aggreation(in_channels = channels*4, out_channels = channels)
        self.aggreation3_mas = Aggreation(in_channels = channels*4, out_channels = channels)  
        
        ##Stage4
        self.spp_img = SPP(in_channels = channels, out_channels = channels, num_layers = 4, interpolation_type = 'bicubic')
        self.spp_mas = SPP(in_channels = channels, out_channels = channels, num_layers = 4, interpolation_type = 'bicubic')
        
        self.block4_1 = nn.Conv2d(in_channels = channels, out_channels = 3, kernel_size = 1, stride = 1)
        self.block4_2 = nn.Conv2d(in_channels = channels, out_channels = 1, kernel_size = 1, stride = 1)
          
    def forward(self, x, mask, mask_d, mask_e):
        
        out_backbone = self.backbone(x)    
        out = self.fusion(out_backbone)
        
        ##Stage0
        out0_1, _, _ = self.block0_1(out, mask, mask_d, mask_e)
        out0_2, _, _ = self.block0_2(out0_1, mask, mask_d, mask_e)
        
        agg0_rgb = self.aggreation0_rgb(torch.cat((out0_1, out0_2), dim = 1))

        ##Stage1
        out1_1, _, _ = self.block1_1(agg0_rgb, mask, mask_d, mask_e)
        out1_2, _, _ = self.block1_2(out1_1, mask, mask_d, mask_e)
        
        agg1_rgb = self.aggreation1_rgb(torch.cat((agg0_rgb, out1_1, out1_2), dim = 1))

        ##Stage2
        out2_1, _, _ = self.block2_1(agg1_rgb, mask, mask_d, mask_e)
        out2_2, _, _ = self.block2_2(out2_1, mask, mask_d, mask_e)
        
        agg2_rgb = self.aggreation2_rgb(torch.cat((agg1_rgb, out2_1, out2_2), dim = 1))
        
        ##Stage3
        out3_1, _, _ = self.block3_1(agg2_rgb, mask, mask_d, mask_e)
        out3_2, _, _ = self.block3_2(out3_1, mask, mask_d, mask_e)

        agg3_rgb = self.aggreation3_rgb(torch.cat((agg1_rgb, agg2_rgb, out3_1, out3_2), dim = 1))

        ##Stage4
        spp_rgb = self.spp_img(agg3_rgb)
        out_rgb = self.block4_1(spp_rgb)

        return out_rgb


