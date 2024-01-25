
from __future__ import absolute_import, division, print_function
from pip._vendor.distlib.compat import raw_input

import os
import argparse
import datetime
import yaml
import shutil

file_dir = os.path.dirname(__file__)  # the directory that options.py resides i


## Options for the experiments

# Options for training semantic segmentation and weather classification network
# Various hyper-parameters for tuning


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def _dataset_options(self):
        # Model Options
        self.parser.add_argument("--data_root", type=str, default='./dataset',
                                 help="path to Dataset ")
        self.parser.add_argument("--dataset", type=str, default='cityscapes',
                                 choices=['cityscapes', 'city_lost', 'kitti_2015', 'sceneflow', 'kitti_mix', 'acdc', 'acdc_city'],
                                 help='Name of dataset')
        self.parser.add_argument("--num_classes", type=int, default=None,
                                 help="num classes (default: auto)")

    def _model_options(self):
        # Deeplab Options
        self.parser.add_argument("--model", type=str, default='resnet18',
                                 choices=['resnet18',  'mobilenetv2', 'resnet34',
                                          'efficientnetb0'], help='model name')
        self.parser.add_argument("--separable_conv", action='store_true', default=False,
                                 help="apply separable conv to decoder and aspp")
        self.parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    def _train_options(self):
        # Train Options
        self._train_learning_options()
        self._train_size_options()
        self._train_print_options()
        self._train_resume_options()
        self._validate_options()

    def _train_learning_options(self):
        self.parser.add_argument('--epochs', type=int, default=200, metavar='N',
                                 help='number of epochs to train (default: auto)')
        self.parser.add_argument('--start_epoch', type=int, default=5,
                                 metavar='N', help='start epochs (default:0)')
        self.parser.add_argument("--total_itrs", type=int, default=30e3,
                                 help="epoch number (default: 30k)")

        self.parser.add_argument("--lr", type=float, default=5e-4,
                                 help="learning rate (default: 0.001, disparity(0.001...), semantic(0.01...))")
        self.parser.add_argument("--last_lr", type=float, default=1e-6,
                                 help="last learning rate, default 1e-6 ")

        self.parser.add_argument("--lr_policy", type=str, default='step',
                                 choices=['poly', 'step', 'cos', 'cos_step', 'cos_annealing'],
                                 help="learning rate scheduler policy")
        self.parser.add_argument("--weight_decay", type=float, default=1e-5,
                                 help='weight decay (default: 1e-5)')
        self.parser.add_argument("--optimizer_policy", type=str, default='ADAM', choices=['SGD', 'ADAM'],
                                 help="learning rate scheduler policy")

        self.parser.add_argument("--epsilon", type=float, default=1e-2,
                                 help='parameter for balancing class weight [1e-2, 2e-2, 5e-2, 1e-1]')

        self.parser.add_argument('--use_balanced_weights', action='store_true', default=True,
                                 help='whether to use balanced weights (default: True)')

    def _train_size_options(self):
        self.parser.add_argument("--batch_size", type=int, default=8,
                                 help='batch size (default: 8)')
        self.parser.add_argument("--val_batch_size", type=int, default=8,
                                 help='batch size for validation (default: 8)')
        self.parser.add_argument("--step_size", type=int, default=10000)
        self.parser.add_argument("--crop_size", type=int, default=384)      # width value, in cityscapes dataset height is divided by 2
        self.parser.add_argument("--img_height", type=int, default=512)
        self.parser.add_argument("--img_width", type=int, default=1024)
        self.parser.add_argument("--val_img_height", type=int, default=1024)
        self.parser.add_argument("--val_img_width", type=int, default=2048)
        self.parser.add_argument('--base-size', type=int, default=1024,
                                 help='base image size')
        self.parser.add_argument("--crop_val", action='store_true', default=False,
                                 help='crop validation (default: False)')

    def _train_print_options(self):
        self.parser.add_argument("--gpu_id", type=str, default='0',
                                 help="GPU ID")
        self.parser.add_argument("--random_seed", type=int, default=1000,
                                 help="random seed (default: 1000)")
        self.parser.add_argument("--print_freq", type=int, default=10,
                                 help="print interval of loss (default: 10)")
        self.parser.add_argument("--summary_freq", type=int, default=40,
                                 help="summary interval of loss (default: 100)")
        self.parser.add_argument("--val_print_freq", type=int, default=10,
                                 help="print interval of validation (default: 10)")
        self.parser.add_argument("--val_save_freq", type=int, default=20,
                                 help="print interval of validation (default: 20)")
        self.parser.add_argument("--val_interval", type=int, default=100,
                                 help="epoch interval for eval (default: 100)")
        self.parser.add_argument("--download", action='store_true', default=False,
                                 help="download datasets")

        # Log
        self.parser.add_argument('--no_build_summary', action='store_true',
                            help='Dont save sammary when training to save space')
        self.parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Save checkpoint frequency (epochs)')

    def _train_resume_options(self):
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='put the path to resuming file if needed')
        self.parser.add_argument("--continue_training", action='store_true', default=False)
        self.parser.add_argument("--transfer_disparity", action='store_true', default=False)
        self.parser.add_argument('--checkname', type=str, default='test',
                                 help='set the checkpoint name')

    def _validate_options(self):
        self.parser.add_argument("--test_only", action='store_true', default=False)


    def _pascal_voc_options(self):
        # PASCAL VOC Options
        self.parser.add_argument("--year", type=str, default='2012',
                                 choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')


    def _depth_options(self):
        self.parser.add_argument('--use_depth', action="store_true", default=False,
                                 help='training with depth image or not (default: False)')

    def _stereo_depth_prediction_options(self):

        self.parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
        self.parser.add_argument('--train_disparity', action='store_true', help='train_disparity with segmentation')
        self.parser.add_argument('--train_semantic', action='store_true', help='train segmentation')
        self.parser.add_argument('--with_refine', action='store_true', help='train segmentation')
        self.parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')
        self.parser.add_argument('--feature_similarity', default='correlation', type=str,
                            help='Similarity measure for matching cost')

        # Loss
        self.parser.add_argument('--highest_loss_only', action='store_true',
                            help='Only use loss on highest scale for finetuning')
        self.parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')
        self.parser.add_argument('--with_depth_level_loss', action='store_true', help='train segmentation')

        # md-fusion
        self.parser.add_argument('--not_md_fusion', action='store_true', help='not apply md_fusion')
        self.parser.add_argument('--without_balancing', action='store_true', help='not apply balancing')
        self.parser.add_argument('--without_class_balancing', action='store_true', help='not apply balancing')
        self.parser.add_argument('--without_semantic_border', action='store_true', help='not apply balancing')

        # output_dir for test
        self.parser.add_argument('--output_dir', default='output', type=str,
                                 help='Directory to save inference results')

        self.parser.add_argument("--new_crop", action='store_true', default=False)
        self.parser.add_argument("--disp_to_obst_ch", action='store_true', default=False)

    def _train_hyper_parameters(self):
        self.parser.add_argument('--amp', action='store_true', default=False)

        self.parser.add_argument("--sem_weight", type=float, default=1e-1,
                                 help='parameter for balancing class weight [1e-2, 2e-2, 5e-2, 1e-1]')
        self.parser.add_argument("--weather_weight", type=float, default=1e-1,
                                 help='parameter for balancing class weight [1e-2, 2e-2, 5e-2, 1e-1]')
        self.parser.add_argument("--disp_weight", type=float, default=1e-1,
                                 help='parameter for balancing class weight [1e-2, 2e-2, 5e-2, 1e-1]')
        self.parser.add_argument("--pseudo_disp_weight", type=float, default=1e-1,
                                 help='parameter for balancing class weight [1e-2, 2e-2, 5e-2, 1e-1]')

        self.parser.add_argument("--debug", action='store_true', default=False)
        self.parser.add_argument("--acdc_cityfull", action='store_true', default=False)
        self.parser.add_argument("--use_gamma_correction", action='store_true', default=False)

        self.parser.add_argument("--save_val_results", action='store_true', default=False,
                                 help="save segmentation results to \"./results\"")
        self.parser.add_argument("--save_each_results", action='store_true', default=False)


    def _spade_base_options(self):
        self.parser.add_argument("--use_SPADE", action='store_true', default=False)

        # for generator
        self.parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3',
                            help='instance normalization or batch normalization')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')
        self.parser.add_argument('--z_dim', type=int, default=256,
                            help="dimension of the latent z vector")
        self.parser.add_argument('--aspect_ratio', type=float, default=2.0,
                            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        self.parser.add_argument('--spade_crop_size', type=int, default=1024,
                            help='Crop to the width of crop_size (after initially scaling the images to load_size.)')

        # for discriminator
        self.parser.add_argument('--norm_D', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')


        self.parser.add_argument('--norm_E', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')

        self.parser.add_argument('--label_nc', type=int, default=19,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='more',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true',
                            help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--contain_dontcare_label', action='store_true',
                            help='if the label map contains dontcare label (dontcare=255)')
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        self.parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

        # load pretrained-models
        self.parser.add_argument('--resume_SPADE', action='store_true', default=False)
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--spade_checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--load_pretrained_SPADE_name', type=str, default='cityscapes_train_SPADE_new1', help='load pretrained SPADE')


    def _spade_train_options(self):
        self.parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        self.parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')

        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true',
                            help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true',
                            help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        self.parser.add_argument('--lambda_kld', type=float, default=0.05)

        self.parser.add_argument('--niter', type=int, default=50,
                            help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        self.parser.add_argument('--niter_decay', type=int, default=0,
                            help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

        self.parser.add_argument("--print_spade_freq", type=int, default=100,
                                 help="print interval of loss (default: 100)")

        self.parser.add_argument("--eval_FID", action='store_true', default=False)

    def _spade_with_sem_options(self):
        self.parser.add_argument("--use_SPADE_with_SEM", action='store_true', default=False)


    def parse(self):
        self._dataset_options()
        self._model_options()
        self._train_options()
        # self._pascal_voc_options()
        self._depth_options()
        self._stereo_depth_prediction_options()
        self._train_hyper_parameters()

        self._spade_base_options()
        self._spade_train_options()

        self._spade_with_sem_options()

        self.options = self.parser.parse_args()


        self.options.semantic_nc = self.options.label_nc + \
            (1 if self.options.contain_dontcare_label else 0) + \
            (0 if self.options.no_instance else 1)


        return self.options
