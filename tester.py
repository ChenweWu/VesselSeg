import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
import os
from PIL import Image
class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, dataset_path, show=False):
        # super(Trainer, self).__init__()
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader
        self.model = nn.DataParallel(model.cuda())
        self.dataset_path = dataset_path
        self.show = show
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.show:
            dir_exists("save_picture")
            remove_files("save_picture")
        cudnn.benchmark = True
        self.DATA_PATH = '/home/ubuntu/arvotest/CHASEDB1/'
    def snow(self,in_image, in_mask, p, channel_avgs=None):
        p = 1-p
        out_img = []
        for i in range(3):
            img = in_image[:,:,i].copy()
            # with open('shape.txt', 'a') as f:
            #     f.write(str(img.shape))
            #     f.write("       aa         ")
            #     f.write(str(in_image.shape))
            #     f.write("       bb         ")            
            mask = in_mask.copy()
            num_pixels = np.sum(mask)
            masko = mask.copy().flatten()
            mask = mask.flatten()
            sample_size = round(p * num_pixels)
            img_copy = img.copy()
            img_flat = img_copy.flatten()


        #  sample = np.random.choice(num_pixels, size=sample_size, replace=False)
            indices = np.where(masko == 1)[0]
            sample = np.random.choice(indices,size=sample_size, replace=False)
            masko[sample]=0

            if channel_avgs is None:
                channel_avgs = np.average(img_flat[mask!=0])
            #img_flat[mask][sample] = [channel_avg for i in range(len(sample))]
            img_flat[masko!=0] = channel_avgs
            # with open('output.txt', 'a') as f:
            #     f.write(str(np.sum(mask))+"aa"+str(np.sum(masko)))
            #     f.write("       aa         ")
            #     f.write(str(channel_avg))
            #     f.write("       bb         ")
            img_flat=img_flat.reshape((512, 512))
            out_img.append(img_flat)
        snow_img = np.dstack(out_img) 
        # with open('output2.txt', 'a') as f:
        #         f.write(str(len(out_img))+" "+str(out_img[0].shape))
        #         f.write("Len")
        #         f.write(str(snow_img.shape))
        #         f.write("       aa         ")
        #         f.write(str(channel_avg))
        #         f.write("       bb         ")
        return snow_img
    def test(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for i, (img, gt,file_path) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                pre = self.model(img)
              #  loss = self.loss(pre, gt)
             #   self.total_loss.update(loss.item())
              #  self.batch_time.update(time.time() - tic)

                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 512, 512
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300

                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)
                # img = img[0,0,...]
                # gt = gt[0,0,...]
                pre = pre[0,0,...]
                if self.show:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= self.CFG.threshold, 1, 0)
                    
                    img = np.array(Image.open(os.path.join(self.DATA_PATH, file_path[0][:-4][4:])).convert('RGB'))

                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(
                        f"save_picture/img{file_path}.png", np.uint8(img))
                    # cv2.imwrite(
                    #     f"save_picture/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                    # cv2.imwrite(
                    #     f"save_picture/pre{i}.png", np.uint8(predict*255))
                    # cv2.imwrite(
                    #     f"save_picture/pre_b{i}.png", np.uint8(predict_b*255))
                    # image = np.uint8(img.cpu().numpy()*255)
                    cv2.imwrite(
                        f"save_picture/pre_snow{file_path}.png",np.uint8(self.snow(img,predict_b,0.3)))
                    
