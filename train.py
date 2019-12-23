""" script for training the MSG-GAN on given dataset """

# Set to True if using in SageMaker
USE_SAGEMAKER = False

import argparse
import numpy as np
import torch as th
import datetime
import dateutil.tz
from torch.backends import cudnn
from torchvision.datasets import CIFAR10

# define the device for the training script
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# enable fast training
cudnn.benchmark = True
# set seed = 3
th.manual_seed(seed=3)


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for generator")

    parser.add_argument("--generator_optim_file", action="store", type=str,
                        default=None,
                        help="saved state for generator optimizer")

    parser.add_argument("--shadow_generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for the shadow generator")

    parser.add_argument("--discriminator_file", action="store", type=str,
                        default=None,
                        help="pretrained_weights file for discriminator")

    parser.add_argument("--discriminator_optim_file", action="store", type=str,
                        default=None,
                        help="saved state for discriminator optimizer")

    parser.add_argument("--dataset_dir", action="store", type=str,
                        default="../../../../Datasets/cifar10",
                        help="directory to download/use a pytorch dataset")

    parser.add_argument("--pytorch_dataset", action="store", type=str,
                        default=None,
                        help="Whether to use a default pytorch dataset" +
                             "Currently supported:" +
                             "1.) cifar10")

    parser.add_argument("--images_dir", action="store", type=str,
                        default="../../../../Datasets/102flowers/jpg",
                        help="path for the images directory")

    parser.add_argument("--folder_distributed", action="store", type=bool,
                        default=False,
                        help="whether the images directory contains folders or not")

    parser.add_argument("--flip_augment", action="store", type=bool,
                        default=True,
                        help="whether to randomly mirror the images during training")

    parser.add_argument("--sample_dir", action="store", type=str,
                        default=None,
                        help="path for the generated samples directory")

    parser.add_argument("--output_dir", action="store", type=str,
                        default=None,
                        help="path for saved models directory")

    parser.add_argument("--log_dir", action="store", type=str,
                        default=None,
                        help="path for the log file directory")

    parser.add_argument("--loss_function", action="store", type=str,
                        default="relativistic-hinge",
                        help="loss function to be used: " +
                             "standard-gan, wgan-gp, lsgan," +
                             "lsgan-sigmoid," +
                             "hinge, relativistic-hinge")

    parser.add_argument("--depth", action="store", type=int,
                        default=6,
                        help="Depth of the GAN")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=512,
                        help="latent size for the generator")

    parser.add_argument("--batch_size", action="store", type=int,
                        default=32,
                        help="batch_size for training")

    parser.add_argument("--start", action="store", type=int,
                        default=1,
                        help="starting epoch number")

    parser.add_argument("--num_epochs", action="store", type=int,
                        default=2000,
                        help="number of epochs for training")

    parser.add_argument("--feedback_factor", action="store", type=int,
                        default=5,
                        help="number of logs to generate per epoch")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=25,
                        help="number of samples to generate for creating the grid" +
                             " should be a square number preferably")

    parser.add_argument("--checkpoint_factor", action="store", type=int,
                        default=20,
                        help="save model per n epochs")

    parser.add_argument("--g_lr", action="store", type=float,
                        default=0.003,
                        help="learning rate for generator")

    parser.add_argument("--d_lr", action="store", type=float,
                        default=0.003,
                        help="learning rate for discriminator")

    parser.add_argument("--adam_beta1", action="store", type=float,
                        default=0,
                        help="value of beta_1 for adam optimizer")

    parser.add_argument("--adam_beta2", action="store", type=float,
                        default=0.99,
                        help="value of beta_2 for adam optimizer")

    parser.add_argument("--use_eql", action="store", type=bool,
                        default=True,
                        help="Whether to use equalized learning rate or not")

    parser.add_argument("--use_ema", action="store", type=bool,
                        default=True,
                        help="Whether to use exponential moving averages or not")

    parser.add_argument("--ema_decay", action="store", type=float,
                        default=0.999,
                        help="decay value for the ema")

    parser.add_argument("--data_percentage", action="store", type=float,
                        default=100,
                        help="percentage of data to use")

    parser.add_argument("--num_workers", action="store", type=int,
                        default=3,
                        help="number of parallel workers for reading files")

    args = parser.parse_args()
    print('args={}'.format(args))

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    from MSG_GAN.GAN import MSG_GAN
    from data_processing.DataLoader import FlatDirectoryImageDataset, \
        get_transform, get_data_loader, FoldersDistributedDataset, IgnoreLabels
    from MSG_GAN import Losses as lses

    # create a data source:
    if args.pytorch_dataset is None:
        data_source = FlatDirectoryImageDataset if not args.folder_distributed \
            else FoldersDistributedDataset

        dataset = data_source(
            args.images_dir,
            transform=get_transform((int(np.power(2, args.depth + 1)),
                                     int(np.power(2, args.depth + 1))),
                                    flip_horizontal=args.flip_augment))
    else:
        dataset_name = args.pytorch_dataset.lower()
        if dataset_name == "cifar10":
            dataset = IgnoreLabels(CIFAR10(args.dataset_dir,
                                           transform=get_transform((int(np.power(2, args.depth + 1)),
                                                                    int(np.power(2, args.depth + 1))),
                                                                   flip_horizontal=args.flip_augment),
                                           download=True))
        else:
            raise Exception("Unknown dataset requested")

    data = get_data_loader(dataset, args.batch_size, args.num_workers)
    print("Total number of images in the dataset:", len(dataset))

    # create a gan from these
    msg_gan = MSG_GAN(depth=args.depth,
                      latent_size=args.latent_size,
                      use_eql=args.use_eql,
                      use_ema=args.use_ema,
                      ema_decay=args.ema_decay,
                      device=device)

    if args.generator_file is not None:
        # load the weights into generator
        print("loading generator_weights from:", args.generator_file)
        msg_gan.gen.load_state_dict(th.load(args.generator_file))

    # print("Generator Configuration: ")
    # print(msg_gan.gen)

    if args.shadow_generator_file is not None:
        # load the weights into generator
        print("loading shadow_generator_weights from:",
              args.shadow_generator_file)
        msg_gan.gen_shadow.load_state_dict(
            th.load(args.shadow_generator_file))

    if args.discriminator_file is not None:
        # load the weights into discriminator
        print("loading discriminator_weights from:", args.discriminator_file)
        msg_gan.dis.load_state_dict(th.load(args.discriminator_file))

    # print("Discriminator Configuration: ")
    # print(msg_gan.dis)

    # create optimizer for generator:
    gen_params = [{'params': msg_gan.gen.style.parameters(), 'lr': args.g_lr * 0.01, 'mult': 0.01},
                  {'params': msg_gan.gen.layers.parameters(), 'lr': args.g_lr},
                  {'params': msg_gan.gen.rgb_converters.parameters(), 'lr': args.g_lr}]
    gen_optim = th.optim.Adam(gen_params, args.g_lr,
                              [args.adam_beta1, args.adam_beta2])

    dis_optim = th.optim.Adam(msg_gan.dis.parameters(), args.d_lr,
                              [args.adam_beta1, args.adam_beta2])

    if args.generator_optim_file is not None:
        print("loading gen_optim_state from:", args.generator_optim_file)
        gen_optim.load_state_dict(th.load(args.generator_optim_file))

    if args.discriminator_optim_file is not None:
        print("loading dis_optim_state from:", args.discriminator_optim_file)
        dis_optim.load_state_dict(th.load(args.discriminator_optim_file))

    loss_name = args.loss_function.lower()

    if loss_name == "hinge":
        loss = lses.HingeGAN
    elif loss_name == "relativistic-hinge":
        loss = lses.RelativisticAverageHingeGAN
    elif loss_name == "standard-gan":
        loss = lses.StandardGAN
    elif loss_name == "lsgan":
        loss = lses.LSGAN
    elif loss_name == "lsgan-sigmoid":
        loss = lses.LSGAN_SIGMOID
    elif loss_name == "wgan-gp":
        loss = lses.WGAN_GP
    else:
        raise Exception("Unknown loss function requested")

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    if args.pytorch_dataset is not None:
        dataName = 'cifar'
    elif args.images_dir.find('celeb') != -1:
        dataName = 'celeb'
    else:
        dataName = 'flowers'
    output_dir = 'output/%s_%s_%s' % \
        ('attnmsggan', dataName, timestamp)
    args.model_dir = output_dir + '/models'
    args.sample_dir = output_dir + '/images'
    args.log_dir = output_dir + '/logs'

    # train the GAN
    msg_gan.train(
        data,
        gen_optim,
        dis_optim,
        loss_fn=loss(msg_gan.dis),
        num_epochs=args.num_epochs,
        checkpoint_factor=args.checkpoint_factor,
        data_percentage=args.data_percentage,
        feedback_factor=args.feedback_factor,
        num_samples=args.num_samples,
        sample_dir=args.sample_dir,
        save_dir=args.model_dir,
        log_dir=args.log_dir,
        start=args.start
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
