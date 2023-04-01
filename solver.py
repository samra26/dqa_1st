import torch
from torch.nn import functional as F
from dqa import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import os
#from ptflops import get_model_complexity_info
size_coarse = (10, 10)
#from pthflops import count_ops
def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net = build_model()
        #self.net.eval()

        
        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'DQA Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        param_size = 0

        for p in model.parameters():
            param_size += p.nelement() * p.element_size()
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        #print(name)
        #print(model)
        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.6f}MB'.format(size_all_mb))
        print("The number of trainable parameters: {:.6f}".format(num_params_t))
        print("The number of parameters: {:.6f}".format(num_params))


    def test(self):
        print('Testing...')

        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():

                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                # start time
                torch.cuda.synchronize()
                tsince = int(round(time.time()*1000)) 
                preds = self.net(images,depth)
                torch.cuda.synchronize()
                ttime_elapsed = int(round(time.time()*1000)) - tsince
                print ('test time elapsed {}ms'.format(ttime_elapsed))
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + 'dqa.png')
                cv2.imwrite(filename, multi_fuse)
      
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        
        loss_vals=  []
        
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label'], data_batch['sal_edge']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label, sal_edge= sal_image.to(device), sal_depth.to(device), sal_label.to(device),sal_edge.to(device)

               
                self.optimizer.zero_grad()
              
                sal_final= self.net(sal_image,sal_depth)

                sal_loss =  dice_loss(sal_final, sal_edge)
               
                r_sal_loss += sal_loss.data
                r_sal_loss_item+=sal_loss.item() * sal_image.size(0)
                sal_loss.backward()
                self.optimizer.step()

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %0.4f  ||sal_final:%0.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss,sal_final_loss ))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)


                    r_sal_loss = 0

                    
                    fsal = sal_final[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_final', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)



            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
        

