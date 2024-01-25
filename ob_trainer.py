import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import datetime
import utils
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt

# from network.warp import disp_warp
from metrics.disparity_metric import d1_metric, thres_metric
from utils.init_trainer import InitOpts
from utils.loss import SegmentationLosses, DisparityLosses, get_smooth_loss
from network.utils import upsample
import scipy
import tensorflow as tf
from collections import OrderedDict
from utils.spade_util import tensor2label, tensor2im

from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


## Trainer for the experiments

# Train and validate code for semantic segmentation and weather classification network


class Trainer(InitOpts):
    def __init__(self, options):
        super().__init__(options)

    def train(self):
        interval_loss,  train_epoch_loss = 0.0, 0.0
        print_cycle, data_cycle = 0.0, 0.0
        step = 0

        # empty the cache
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

        # switch to train mode
        self.model.train()
        num_img_tr = len(self.train_loader)

        if self.opts.train_semantic:
            self.criterion.step_counter = 0
            # Learning rate summary
            base_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('base_lr', base_lr, self.cur_epochs)
            self.evaluator.reset()


        last_data_time = time.time()
        for i, sample in enumerate(self.train_loader):
            data_loader_time = time.time() - last_data_time
            data_cycle += data_loader_time
            self.num_iter += 1
            model_start_time = time.time()

            left = sample['left'].to(self.device, dtype=torch.float32)

            if 'label' in sample.keys():
                labels = sample['label'].to(self.device, dtype=torch.long)

            if 'weather' in sample.keys():
                gt_weather = sample['weather'].to(self.device)

            if self.opts.train_semantic:
                self.optimizer.zero_grad()

                left_seg, weather_pred = self.model(left)
                loss = self.criterion(left_seg, labels, sample)
                loss_weather = F.cross_entropy(weather_pred, gt_weather.view(-1))
                total_loss = (loss * self.opts.sem_weight + loss_weather * self.opts.weather_weight)
                # total_loss = (loss_weather * self.opts.weather_weight)

                interval_loss += total_loss
                train_epoch_loss += total_loss

                total_loss.backward()
                self.optimizer.step()

                one_cycle_time = time.time() - model_start_time
                print_cycle += one_cycle_time

                if self.num_iter % self.opts.print_freq == 0:
                    interval_loss = interval_loss / self.opts.print_freq
                    print("Epoch: [%3d/%3d] Itrs: [%5d/%5d] dataloader time : %4.2fs training time: %4.2fs time_per_img: %4.2fs Loss=%f" %
                          (self.cur_epochs, self.opts.epochs, i, num_img_tr, data_cycle, print_cycle,
                           print_cycle/self.opts.print_freq/self.opts.batch_size, interval_loss))
                    self.writer.add_scalar('train/total_loss_print_freq', interval_loss, self.num_iter)
                    interval_loss, print_cycle, data_cycle  = 0.0, 0.0, 0.0

                if self.num_iter % self.opts.summary_freq == 0:
                    summary_time_start = time.time()
                    self.writer.add_scalar('train/total_loss_summary_freq', total_loss.item(), self.num_iter)
                    self.writer.add_scalar('train/sem_loss_summary_freq', loss.item(), self.num_iter)
                    self.writer.add_scalar('train/weather_loss_summary_freq', loss_weather.item(), self.num_iter)
                    summary_time = time.time() - summary_time_start
                    print("summary_time : {}".format(summary_time))
                last_data_time = time.time()

                del total_loss, sample

            if self.opts.use_SPADE:
                time1 = time.time()
                # run_generator_one_step
                left_image = left / 255.
                left_image -= 0.5
                left_image /= 0.5

                self.optimizer_G.zero_grad()
                g_losses, generated = self.model.compute_generator_loss(labels, left_image)
                g_loss = sum(g_losses.values()).mean()
                g_loss.backward()
                self.optimizer_G.step()
                self.g_losses = g_losses
                self.generated = generated

                # run_discriminator_one_step
                self.optimizer_D.zero_grad()
                d_losses = self.model.compute_discriminator_loss(labels, left_image)
                d_loss = sum(d_losses.values()).mean()
                d_loss.backward()
                self.optimizer_D.step()
                self.d_losses = d_losses
                time2 = time.time()

                if i % self.opts.print_spade_freq == 0:
                    ## print current training situations
                    losses = {**self.g_losses, **self.d_losses}
                    self.print_current_errors(self.cur_epochs, i, losses , (time2-time1)/self.opts.batch_size)
                    step = self.cur_epochs * len(self.train_loader) + i
                    self.plot_current_errors(losses, step)

                    ## display current training results
                    visuals = OrderedDict([('input_label', labels.unsqueeze(1)),
                                           ('synthesized_image', generated),
                                           ('real_image', left_image)])
                    self.display_current_results(visuals, self.cur_epochs, step)

        if self.opts.train_semantic:
            train_epoch_loss = train_epoch_loss / num_img_tr
            self.writer.add_scalar('train/total_loss_epoch', train_epoch_loss, self.cur_epochs)


    def validate(self):
        """Do validation and return specified samples"""
        print("validation...")
        if self.opts.train_semantic:
            self.evaluator.reset()
        if self.opts.eval_FID:
            dims = 2048
            pred_real_fid_arr = np.empty((len(self.val_loader), dims))
            pred_fake_fid_arr = np.empty((len(self.val_loader), dims))
            start_fid_idx = 0

        self.time_val = []

        # empty the cache to infer in high res
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        # switch to evaluate mode
        self.model.eval()

        valid_samples, scores, img_id = 0, 0, 0
        num_val = len(self.val_loader)

        with torch.no_grad():
            start = time.time()
            for i, sample in enumerate(self.val_loader):
                data_time = time.time() - start
                self.time_val_dataloader.append(data_time)

                left = sample['left'].to(self.device, dtype=torch.float32)

                if 'label' in sample.keys():
                    labels = sample['label']

                if 'weather' in sample.keys():
                    gt_weather = sample['weather'].to(self.device)

                if 'input_semantics' in sample.keys():
                    label_map = sample['input_semantics'].to(self.device)
                if 'real_image' in sample.keys():
                    real_image = sample['real_image'].to(self.device)

                valid_samples += 1
                start_time = time.time()

                if self.opts.train_semantic:
                    left_seg, pred_weather = self.model(left)
                    fwt = time.time() - start_time

                    preds = left_seg.detach().max(dim=1)[1].cpu().numpy()
                    targets = labels.numpy()
                    gt_weather = gt_weather.view(-1)
                    self.evaluator.add_batch_weather(gt_weather, pred_weather)
                    self.evaluator.add_batch(targets, preds, gt_weather.cpu().numpy())

                    # first batch stucked on some process.. --> time cost is wierd on i==0
                    if i != 0:
                        self.time_val.append(fwt)
                        if i % self.opts.val_print_freq == 0:
                            # check validation fps
                            print(
                                "[%d/%d] Model passed time (bath size=%d): %.3f (Mean time per img: %.3f), Dataloader time : %.3f" % (
                                    i, num_val,
                                    self.opts.val_batch_size, fwt,
                                    sum(self.time_val) / len(self.time_val) / self.opts.val_batch_size, data_time))

                if self.opts.use_SPADE:
                    z = None
                    if self.opts.use_vae:
                        real_image_ = real_image / 255.
                        real_image_ -= 0.5
                        real_image_ /= 0.5

                        mu, logvar = self.model.netE(real_image_)
                        z = self.reparameterize(mu, logvar)

                    if self.opts.use_SPADE_with_SEM:
                        # create one-hot label map from predicted semantic segmentations
                        pred_segs = left_seg.clone()
                        pred_segs = upsample(pred_segs, real_image.shape[2:])

                        label_map = pred_segs.max(dim=1)[1].unsqueeze(1)
                        bs, _, h, w = label_map.size()
                        nc = self.opts.label_nc + 1 if self.opts.contain_dontcare_label \
                            else self.opts.label_nc
                        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_().to(self.device)
                        input_semantics = input_label.scatter_(1, label_map, 1.0)

                        fake_image = self.model.netG(input_semantics, z=z)
                    else:
                        label_map[label_map == 255] = self.opts.label_nc
                        label_map = label_map.unsqueeze(1)
                        bs, _, h, w = label_map.size()
                        nc = self.opts.label_nc + 1 if self.opts.contain_dontcare_label \
                            else self.opts.label_nc
                        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_().to(self.device)
                        input_semantics = input_label.scatter_(1, label_map, 1.0)
                        fake_image = self.model.netG(input_semantics, z=z)

                    if self.opts.eval_FID:
                        fid_start = time.time()
                        # get activation
                        real_image_ = real_image / 255.
                        real_fid = self.model.FID(real_image_)[0]
                        fake_image_ = (fake_image + 1)/2.0
                        fake_fid = self.model.FID(fake_image_)[0]

                        if real_fid.size(2) != 1 or real_fid.size(3) != 1:
                            real_fid = adaptive_avg_pool2d(real_fid, output_size=(1, 1))
                        if fake_fid.size(2) != 1 or fake_fid.size(3) != 1:
                            fake_fid = adaptive_avg_pool2d(fake_fid, output_size=(1, 1))

                        real_fid = real_fid.squeeze(3).squeeze(2).cpu().numpy()
                        fake_fid = fake_fid.squeeze(3).squeeze(2).cpu().numpy()

                        pred_real_fid_arr[start_fid_idx:start_fid_idx + real_fid.shape[0]] = real_fid
                        pred_fake_fid_arr[start_fid_idx:start_fid_idx + fake_fid.shape[0]] = fake_fid

                        start_fid_idx = start_fid_idx + real_fid.shape[0]
                        fid_end = time.time()
                        if (i % 20 ==0):
                            print("{} th FID's are calculated : {} sec".format(i, fid_end-fid_start))

                if self.opts.save_val_results and (i % self.opts.val_save_freq == 0):
                    # save all validation results images
                    if self.opts.train_semantic:
                        if self.opts.use_SPADE_with_SEM:
                            self.save_valid_img_in_results(left, targets, preds, i, fake_image=fake_image)
                        else:
                            self.save_valid_img_in_results(left, targets, preds, i)
                    elif self.opts.use_SPADE:
                        self.save_valid_img_SPADE_only(real_image, label_map.detach().cpu().numpy(), i, fake_image=fake_image)

                    img_id += 1

                start = time.time()
            del sample

        # test validation performance of semantic segmentation
        if self.opts.train_semantic:
            score = self.evaluator.get_results()
            save_filename = self.saver.save_file_return()
            weather_acc = self.evaluator.get_weather_results(save_filename)
            self.performance_test(score, weather_acc, save_filename)


        # test validation performance of GAN
        if self.opts.eval_FID:
            # calculate FID score
            fid_value = self.calculate_frechet_distance(pred_real_fid_arr, pred_fake_fid_arr)

            print('FID: {}'.format(fid_value))
            with open(self.saver.save_file_return(), 'a') as f:
                f.write("FID : {}\n".format(fid_value))


        if not self.opts.test_only:
            if self.opts.train_semantic:
                self.save_checkpoints_sem(score)

                if self.opts.dataset != 'kitti_mix':
                    if score['Mean IoU'] > self.best_score:  # save best model
                        self.best_score = score['Mean IoU']
                        self.best_score_epoch = self.cur_epochs
                        self.save_checkpoints_sem(score, is_best=True, best_type='score')
                    print('\nbest score epoch: {}, best score: {}'.format(self.best_score_epoch, self.best_score))

            if self.opts.use_SPADE:
                # save models
                self.save_spade_weight('latest')

                if self.opts.eval_FID and fid_value < self.best_fid_value:
                    self.best_fid_value = fid_value
                    self.best_fid_epoch = self.cur_epochs
                    self.save_spade_weight('best')
                print('best FID values:{} at {} epoch'.format(self.best_fid_value, self.best_fid_epoch))
                with open(self.saver.save_file_return(), 'a') as f:
                    f.write('best FID values:{} at {} epoch \n'.format(self.best_fid_value, self.best_fid_epoch))

    def test(self):
        self.validate()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def save_checkpoints(self, epe, score, is_best=False, best_type=None):
        if self.n_gpus > 1:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        self.saver.save_checkpoint({
            'epoch': self.cur_epochs,
            "num_iter": self.num_iter,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'epe': epe,
            'score': score,
            'best_score': self.best_score,
            'best_epe': self.best_epe,
            'best_score_epoch': self.best_score_epoch,
            'best_epe_epoch': self.best_epe_epoch
        }, is_best, best_type)

    def save_checkpoints_sem(self, score, is_best=False, best_type=None):
        if self.n_gpus > 1:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        self.saver.save_checkpoint({
            'epoch': self.cur_epochs,
            "num_iter": self.num_iter,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'score': score,
            'best_score': self.best_score,
            'best_score_epoch': self.best_score_epoch,
        }, is_best, best_type)

    def save_spade_weight(self, epoch):
        self.saver.save_spade_checkpoint(self.model.netG, 'G', epoch)
        self.saver.save_spade_checkpoint(self.model.netD, 'D', epoch)
        if self.opts.use_vae:
            self.saver.save_spade_checkpoint(self.model.netE, 'E', epoch)

    def performance_check_train(self, disp_loss, total_loss, pred_disp, gt_disp, mask, score):
        if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
            self.writer.add_scalar('train/mIoU', score["Mean IoU"], self.num_iter)
            self.writer.add_scalar('train/OverallAcc', score["Overall Acc"], self.num_iter)
            self.writer.add_scalar('train/MeanAcc', score["Mean Acc"], self.num_iter)
            self.writer.add_scalar('train/fwIoU', score["FreqW Acc"], self.num_iter)

        if self.opts.train_disparity:
            # pred_disp = pred_disp.squeeze(1)

            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            self.writer.add_scalar('train/epe', epe.item(), self.num_iter)
            self.writer.add_scalar('train/disp_loss', disp_loss.item(), self.num_iter)
            self.writer.add_scalar('train/total_loss', total_loss.item(), self.num_iter)

            d1 = d1_metric(pred_disp, gt_disp, mask)
            self.writer.add_scalar('train/d1', d1.item(), self.num_iter)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
            thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
            self.writer.add_scalar('train/thres1', thres1.item(), self.num_iter)
            self.writer.add_scalar('train/thres2', thres2.item(), self.num_iter)
            self.writer.add_scalar('train/thres3', thres3.item(), self.num_iter)

    def performance_test(self, val_score, weather_acc, save_filename):
        print('Validation:')
        print('[Epoch: %d]' % (self.cur_epochs))

        if self.opts.train_semantic and self.opts.dataset != 'kitti_mix':
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union(save_filename)
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            weather_mIoU = self.evaluator.Mean_Intersection_over_Union_each_weather(save_filename)

            if not self.opts.test_only:
                self.writer.add_scalar('val/mIoU', mIoU, self.cur_epochs)
                self.writer.add_scalar('val/Acc', Acc, self.cur_epochs)
                self.writer.add_scalar('val/Acc_class', Acc_class, self.cur_epochs)
                self.writer.add_scalar('val/fwIoU', FWIoU, self.cur_epochs)
                self.writer.add_scalar('val/Acc_weather', weather_acc, self.cur_epochs)

                for key, value in self.val_dst.weather_dict.items():
                    self.writer.add_scalar('val/mIoU_' + key, weather_mIoU[str(value)], self.cur_epochs)

            print(self.evaluator.to_str(val_score))
        else:
            mIoU, Acc, Acc_class, FWIoU = 0, 0, 0, 0

        self.saver.save_val_results_semantic(self.cur_epochs, mIoU, Acc)
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, // weather_acc:{}".format(Acc, Acc_class, mIoU, FWIoU, weather_acc))


    def make_directory(self, root, folders):
        if not os.path.exists(os.path.join(root, folders)):
            os.mkdir(os.path.join(root, folders))

    def save_valid_img_in_results(self, left, targets, preds, img_id, fake_image=None):
        save_start = time.time()
        if not os.path.exists(os.path.join(self.saver.experiment_dir, 'results')):
            os.mkdir(os.path.join(self.saver.experiment_dir, 'results'))

        root_dir = os.path.join(self.saver.experiment_dir, 'results')
        if self.opts.save_each_results:
            self.make_directory(root_dir, 'left_image')
            self.make_directory(root_dir, 'gt_sem')
            self.make_directory(root_dir, 'pred_sem')
            self.make_directory(root_dir, 'overlay')
            if fake_image is not None:
                self.make_directory(root_dir, 'fake_image')
        else:
            self.make_directory(root_dir, 'overall')

        # for i in range(len(left)):
        i = 0
        image = left[i].detach().cpu().numpy()
        # image = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).transpose(1, 2, 0).astype(np.uint8)
        image_ = image.copy()
        image_ = scipy.misc.imresize(image_, (512, 1024))
        image = Image.fromarray(image)

        if self.opts.dataset == 'kitti_2015':
            image = image.crop((0, 8, 1242, 8 + 375))

        target = targets[i]
        target = self.val_loader.dataset.decode_target(target).astype(np.uint8)
        target_ = target.copy()
        target_ = scipy.misc.imresize(target_, (512, 1024))

        target = Image.fromarray(target)
        if self.opts.dataset == 'kitti_2015':
            target = target.crop((0, 8, 1242, 8 + 375))

        pred = preds[i]
        pred = self.val_loader.dataset.decode_target(pred).astype(np.uint8)
        pred_ = pred.copy()
        pred_ = scipy.misc.imresize(pred_, (512, 1024))
        pred = Image.fromarray(pred)
        if self.opts.dataset == 'kitti_2015':
            pred = pred.crop((0, 8, 1242, 8 + 375))
        overlay = Image.blend(image, pred, alpha=0.7)

        if fake_image is not None:
            generated = fake_image[i].detach().cpu().numpy()
            # generated = (np.transpose(generated, (1, 2, 0)) * 255.0).astype(np.uint8)
            generated = ((np.transpose(generated, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
            generated_ = generated.copy()
            generated = Image.fromarray(generated)

        if self.opts.save_each_results:
            image.save(os.path.join(self.saver.experiment_dir, 'results', 'left_image', '%d_left_image.png' % img_id))
            target.save(
                os.path.join(self.saver.experiment_dir, 'results', 'gt_sem', '%d_gt_sem.png' % img_id))
            pred.save(os.path.join(self.saver.experiment_dir, 'results', 'pred_sem', '%d_pred_sem.png' % img_id))
            overlay.save(os.path.join(self.saver.experiment_dir, 'results', 'overlay', '%d_overlay.png' % img_id))
            if fake_image is not None:
                generated.save(os.path.join(self.saver.experiment_dir, 'results', 'fake_image', '%d_fake_image.png' % img_id))
        else:
            overall_list = [image_, target_, pred_]
            if fake_image is not None:
                overall_list += [generated_]

            store_img = np.concatenate([i.astype(np.uint8) for i in overall_list], axis=0)
            store_img = Image.fromarray(store_img)
            store_img.thumbnail((720, 720))
            store_img.save(os.path.join(self.saver.experiment_dir, 'results', 'overall', '%d_overall.png' % img_id))


        save_end = time.time()
        print(" {}   --- Time for saving images:{}".format(img_id, save_end - save_start))


    def save_valid_img_SPADE_only(self, left, targets, img_id, fake_image=None):
        save_start = time.time()
        if not os.path.exists(os.path.join(self.saver.experiment_dir, 'results')):
            os.mkdir(os.path.join(self.saver.experiment_dir, 'results'))

        root_dir = os.path.join(self.saver.experiment_dir, 'results')
        self.make_directory(root_dir, 'real_image')
        self.make_directory(root_dir, 'gt_sem')
        self.make_directory(root_dir, 'fake_image')

        # for i in range(len(left)):
        i = 0
        image = left[i].detach().cpu().numpy()
        # image = (self.denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
        # image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).transpose(1, 2, 0).astype(np.uint8)
        # image = ((np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
        image = (np.transpose(image, (1, 2, 0))).astype(np.uint8)
        image = Image.fromarray(image)

        target = targets[i]
        target = self.val_loader.dataset.decode_target(target.squeeze(0)).astype(np.uint8)
        target = Image.fromarray(target)

        generated = fake_image[i].detach().cpu().numpy()
        # generated = (np.transpose(generated, (1, 2, 0)) * 255.0).astype(np.uint8)
        generated = ((np.transpose(generated, (1, 2, 0)) + 1) / 2.0 * 255.0).astype(np.uint8)
        generated = Image.fromarray(generated)

        image.save(os.path.join(self.saver.experiment_dir, 'results', 'real_image', '%d.png' % img_id))
        target.save(
            os.path.join(self.saver.experiment_dir, 'results', 'gt_sem', '%d.png' % img_id))
        generated.save(os.path.join(self.saver.experiment_dir, 'results', 'fake_image', '%d.png' % img_id))

        save_end = time.time()
        print(" {}   --- Time for saving images:{}".format(img_id, save_end - save_start))


    def calculate_estimate(self, epoch, iter):
        num_img_tr = len(self.train_loader)
        num_img_val = len(self.val_loader)
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) * \
                       (num_img_tr * self.opts.epochs - (
                               iter + 1 + epoch * num_img_tr))) + \
                   int(self.batch_time_e.avg * num_img_val * (
                           self.opts.epochs - (epoch)))
        return str(datetime.timedelta(seconds=estimate))


    def calculate_frechet_distance(self, pred_real_fid_arr, pred_fake_fid_arr, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.mean(pred_real_fid_arr, axis=0)
        sigma1 = np.cov(pred_real_fid_arr, rowvar=False)

        mu2 = np.mean(pred_fake_fid_arr, axis=0)
        sigma2 = np.cov(pred_fake_fid_arr, rowvar=False)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        save_filename = self.saver.save_file_return()
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(save_filename, "a") as log_file:
            log_file.write('%s\n' % message)

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        for tag, value in errors.items():
            value = value.mean().float()
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)
            # self.writer.add_scalar(tag, value, step)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        # show images in tensorboard output
        img_summaries = []
        for label, image_numpy in visuals.items():
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            if len(image_numpy.shape) >= 4:
                image_numpy = image_numpy[0]
            scipy.misc.toimage(image_numpy).save(s, format="jpeg")
            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0],
                                            width=image_numpy.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag=label, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opts.batch_size > 8
            if 'input_label' == key:
                t = tensor2label(t, self.opts.label_nc + 2, tile=tile)
            elif 'pred_sem' == key:
                t = tensor2label(t, self.opts.label_nc + 2, tile=tile)
            else:
                t = tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals