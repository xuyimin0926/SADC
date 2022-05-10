import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
import random
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
from PIL import ImageOps,Image
from torchgeometry.losses import SSIM
import numpy as np
import math
import kornia
from .SADCNet  import DynConvLayer

class SADCModel(BaseModel):
    def name(self):
        return 'Shadow-Aware Dynamic Convolution Module'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='none')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G', 'perceptual', 'grad', 'cons'] # Perceptual, BCE, GAN Loss
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['input_img','final', 'shadow_mask', 'mask_disc', 'input_gt']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.netG,opt.init_type, opt.init_gain, self.gpu_ids,channels=64)
        #image pixel value range
        self.range_img = (0,1)
        self.loss_per_epoch = 0
        self.ratio = 1.0
        self.activation = {}
        self.cons_coef = 0.0
        self.dilation_K = torch.ones(7, 7).cuda()
        self.erosion_K = torch.ones(7, 7).cuda()

        if self.isTrain:
            self.criterionPerceptual = networks.PerceptualLoss().to(self.device)
            self.BCELoss = torch.nn.BCELoss().to(self.device)
            self.GradLoss = networks.GradientLoss().to(self.device)
            self.ConsLoss = networks.ConsistencyLoss().to(self.device)
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input, train=False):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.input_gt = input['C'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)
        # self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)
        self.mask_d = kornia.morphology.dilation(self.shadow_mask, self.dilation_K)
        self.mask_e = kornia.morphology.erosion(self.shadow_mask, self.erosion_K)
  
    def forward(self):
        self.final = self.netG(self.input_img, self.shadow_mask, self.mask_d, self.mask_e)

    def backward_G(self):
        self.loss_perceptual = self.criterionPerceptual(self.final, self.input_gt)
        self.loss_grad = self.GradLoss(self.final, self.input_gt)
        self.loss_cons = self.cons_coef * self.ConsLoss(self.activation)
        self.loss_G = 100 * (self.loss_perceptual + self.loss_grad + self.loss_cons)
        self.loss_G.backward()
        self.loss_per_epoch += self.loss_G.item()

    def optimize_parameters(self):
        self.forward()
        self.netG.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    
    def get_current_visuals(self):
        nim = self.input_img.shape[0]
        all =[]
        for i in range(0,min(nim,5)):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:], range_img=self.range_img)
                        row.append(im)           
            row=tuple(row)
            row = np.hstack(row)
            all.append(row)      
        all = tuple(all)
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])  
    
    def get_prediction(self,input,is_origin=False):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.gt = input['C'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)  
        # self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)
        self.mask_d = kornia.morphology.dilation(self.shadow_mask, self.dilation_K)
        self.mask_e = kornia.morphology.erosion(self.shadow_mask, self.erosion_K)
        self.forward()
        self.shadow_free = self.final
        RES = dict()
        if is_origin:
            RES['final'] = self.shadow_free
            RES['input'] = self.input_img
            RES['gt'] = input['C']
            RES['mask'] = self.shadow_mask

        else:
            RES['final']= util.tensor2im(self.shadow_free,scale=0, range_img=self.range_img)
            RES['input'] = util.tensor2im(self.input_img, scale=0, range_img=self.range_img)
            RES['gt'] = util.tensor2im(input['C'], scale=0, range_img=self.range_img)
            RES['mask'] = util.tensor2im(self.shadow_mask, scale=0, range_img=self.range_img)

        return  RES

    def register_mask_hook(self):
        for name, module in self.netG.named_modules():
            if isinstance(module, DynConvLayer):
                self.activation[name] = module.register_forward_hook(self.get_activation(name))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = (output[1], output[2].detach())
        return hook

    def update_cons_coef(self, warm_e, total_e):
        if self.cepoch < warm_e:
            self.cons_coef = 0.0
        else:
            self.cons_coef = 1 - math.cos(np.pi/2 * (self.cepoch - warm_e)/(total_e-warm_e))
        print('Update coefficient of consistency loss to {:.2f}'.format(self.cons_coef))
    
    def reset_loss(self):
        self.loss_per_epoch = 0
