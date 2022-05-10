from collections import OrderedDict
from options.train_options import TrainOptions
from options.test_options import  TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from PIL import Image
import visdom
from util.util import sdmkdir
from util import util
import time
import os
from skimage.color import rgb2lab
import numpy as np
import torch

def calculate_rmse(recovered, gt, mask):
    # Transform into lab color space
    recovered_lab = rgb2lab(recovered)
    gt_lab = rgb2lab(gt)
    
    return abs((gt_lab-recovered_lab) * mask).sum(), mask.sum()

test_opt = TestOptions().parse()
model = create_model(test_opt)
model.setup(test_opt)

test_data_loader = CreateDataLoader(test_opt)
test_set = test_data_loader.load_data()
test_save_path = os.path.join(test_opt.checkpoints_dir, 'test')

if not os.path.isdir(test_save_path):
    os.makedirs(test_save_path)

model.eval()
idx = 0

non_shadow_mse = 0
shadow_mse = 0
total_mse = 0
total = 0
total_shadow = 0
total_non_shadow =  0

for i, data in enumerate(test_set):
  with torch.no_grad():
    idx += 1
    visuals = model.get_prediction(data, is_origin=True)
    pred = np.transpose(visuals['final'].squeeze().cpu().detach().float().numpy(), (1,2,0))
    gt = np.transpose(visuals['gt'].squeeze().cpu().detach().float().numpy(), (1,2,0))
    mask = np.transpose(visuals['mask'].squeeze().unsqueeze(0).cpu().detach().float().numpy(), (1,2,0))
    im_name = data['imname'][0].split('.')[0]

    curr_non_shadow_mse, curr_non_shadow = calculate_rmse(pred, gt, 1-mask)
    curr_shadow_mse, curr_shadow = calculate_rmse(pred, gt, mask)
    curr_mse, curr = calculate_rmse(pred, gt, np.ones_like(mask))

    non_shadow_mse += curr_non_shadow_mse
    shadow_mse += curr_shadow_mse
    total_mse += curr_mse

    total_non_shadow += curr_non_shadow
    total_shadow += curr_shadow
    total += curr

    pred = util.tensor2im(visuals['final'], scale=0)
    mask = util.tensor2im(visuals['mask'], scale=0)
    gt = util.tensor2im(visuals['gt'], scale=0)
    ori = util.tensor2im(visuals['input'], scale=0)

    util.save_image(pred, os.path.join(test_save_path, im_name+'pred.png'))

print('S: {shadow_rmse:.2f}, NS: {nonshadow_rmse:.2f}, RMSE: {whole_rmse:.2f}'.format(shadow_rmse=shadow_mse/total_shadow, nonshadow_rmse=non_shadow_mse/total_non_shadow, whole_rmse=total_mse/total))

