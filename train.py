import os
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn

from torch import optim
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from model import *
from utils import *

def train(args):
    mode = args.mode
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    batch_size = args.batch_size
    num_epoch = args.num_epoch

    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

    # custom variable
    val_best_score = 10000

    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    print("mode: %s" % mode)
    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    """ data load """
    train_dataset = Datasets(data_dir=data_dir, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    num_data_train = len(train_dataset)
    num_batch_train = np.ceil(num_data_train / batch_size)

    valid_dataset = Datasets(data_dir=data_dir, mode='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    num_data_valid = len(valid_dataset)
    num_batch_valid = np.ceil(num_data_valid / batch_size)

    """ model & loss & optimizer settings """
    gen = Pix2Pix_3D(in_channels=1, out_channels=1).to(device)
    dis = Discriminator_3D(in_channels=2, out_channels=1).to(device)

    loss_func_gan = nn.BCELoss()
    loss_func_l1 = nn.L1Loss()

    optimG = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    optimD = torch.optim.Adam(dis.parameters(), lr=lr, betas=(beta1, beta2))

    """ optimal settings """
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    st_epoch = 0

    """ training """
    for epoch in range(st_epoch + 1, num_epoch + 1):
        gen.train()
        dis.train()

        loss_G_l1_train = []
        loss_G_gan_train = []
        loss_D_real_train = []
        loss_D_fake_train = []

        for batch, data in enumerate(train_dataloader, 1):
            input = data['t1_img'].to(device)
            label = data['t2_img'].to(device)
            output = gen(input)

            """ Discriminator learning """
            set_requires_grad(dis, True)
            optimD.zero_grad()

            # make input data
            real = torch.cat([input, label], dim=1)
            fake = torch.cat([input, output], dim=1)

            # discriminator prediction
            pred_real = dis(real)
            pred_fake = dis(fake.detach())

            # loss calculation
            loss_D_real = loss_func_gan(pred_real, torch.ones_like(pred_real)) # actual -> 1
            loss_D_fake = loss_func_gan(pred_fake, torch.zeros_like(pred_real)) # fake -> 0
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            # backpropagation
            loss_D.backward()
            optimD.step()

            """ Generator learning """
            set_requires_grad(dis, False)
            optimG.zero_grad()

            # make input data
            fake = torch.cat([input, output], dim=1)

            # generator prediction
            pred_fake = dis(fake)

            # loss calculation
            loss_G_gan = loss_func_gan(pred_fake, torch.ones_like(pred_fake)) # fake -> 1
            loss_G_l1 = loss_func_l1(label, output)
            loss_G = loss_G_gan + 100 * loss_G_l1

            # backpropagation
            loss_G.backward()
            optimG.step()

            loss_G_l1_train += [loss_G_l1.item()]
            loss_G_gan_train += [loss_G_gan.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                  "GEN L1 %.4f | GEN GAN %.4f | "
                  "DISC REAL: %.4f | DISC FAKE: %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_G_l1_train),
                   np.mean(loss_G_gan_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

        writer_train.add_scalar('loss_G_l1', np.mean(loss_G_l1_train), epoch)
        writer_train.add_scalar('loss_G_gan', np.mean(loss_G_gan_train), epoch)
        writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
        writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

        """ validation """
        with torch.no_grad():
            gen.eval()
            dis.eval()

            loss_G_l1_val = []
            loss_G_gan_val = []
            loss_D_real_val = []
            loss_D_fake_val = []

            for batch, data in enumerate(valid_dataloader, 1):
                input = data['t1_img'].to(device)
                label = data['t2_img'].to(device)
                output = gen(input)

                # make input data
                real = torch.cat([input, label], dim=1)
                fake = torch.cat([input, output], dim=1)

                # discriminator prediction
                pred_real = dis(real)
                pred_fake = dis(fake.detach())

                # loss calculation
                loss_D_real = loss_func_gan(pred_real, torch.ones_like(pred_real)) # actual -> 1
                loss_D_fake = loss_func_gan(pred_fake, torch.zeros_like(pred_real)) # fake -> 0

                # make input data
                fake = torch.cat([input, output], dim=1)

                # generator prediction
                pred_fake = dis(fake)

                # loss calculation
                loss_G_gan = loss_func_gan(pred_fake, torch.ones_like(pred_fake)) # fake -> 1
                loss_G_l1 = loss_func_l1(label, output)

                # validation loss
                loss_G_l1_val += [loss_G_l1.item()]
                loss_G_gan_val += [loss_G_gan.item()]
                loss_D_real_val += [loss_D_real.item()]
                loss_D_fake_val += [loss_D_fake.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN L1 %.4f | GEN GAN %.4f | "
                      "DISC REAL: %.4f | DISC FAKE: %.4f" %
                      (epoch, num_epoch, batch, num_batch_valid,
                       np.mean(loss_G_l1_val), np.mean(loss_G_gan_val),
                       np.mean(loss_D_real_val), np.mean(loss_D_fake_val)))

            writer_val.add_scalar('loss_G_l1', np.mean(loss_G_l1_val), epoch)
            writer_val.add_scalar('loss_G_gan', np.mean(loss_G_gan_val), epoch)
            writer_val.add_scalar('loss_D_real', np.mean(loss_D_real_val), epoch)
            writer_val.add_scalar('loss_D_fake', np.mean(loss_D_fake_val), epoch)

        if val_best_score > np.mean(loss_G_l1_val):
            val_best_score = np.mean(loss_G_l1_val)
            save(ckpt_dir=ckpt_dir, netG=gen, epoch=epoch)

    writer_train.close()
    writer_val.close()

def test(args):
    mode = args.mode
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    batch_size = args.batch_size

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    print("mode: %s" % mode)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    """ data load """
    test_dataset = Datasets(data_dir=data_dir, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    num_data_test = len(test_dataset)
    num_batch_test = np.ceil(num_data_test / batch_size)

    """ model & loss & optimizer settings """
    gen = Pix2Pix_3D(in_channels=1, out_channels=1)
    loss_func_l1 = nn.L1Loss()

    gen, _ = load(ckpt_dir=ckpt_dir, netG=gen)
    gen.to(device)

    with torch.no_grad():
        gen.eval()
        loss_G_l1_test = []

        for batch, data in enumerate(test_dataloader, 1):
            input = data['t1_img'].to(device)
            label = data['t2_img'].to(device)
            sub_id = data['sub_id']
            output = gen(input)

            """ loss calculation """
            loss_G_l1 = loss_func_l1(output, label)
            loss_G_l1_test += [loss_G_l1.item()]

            print("TEST: BATCH %04d / %04d | GEN L1 %.4f" %
                  (batch, num_batch_test, np.mean(loss_G_l1_test)))

            for j in range(label.shape[0]):
                input_ = input[j]
                label_ = label[j]
                output_ = output[j]
                sub_id_ = sub_id[j]

                np.save(os.path.join(result_dir, '{}_input.npy'.format(sub_id_)), input_.cpu().detach().numpy())
                np.save(os.path.join(result_dir, '{}_label.npy'.format(sub_id_)), label_.cpu().detach().numpy())
                np.save(os.path.join(result_dir, '{}_output.npy'.format(sub_id_)), output_.cpu().detach().numpy())

        print('AVERAGE TEST: GEN L1 %.4f' % np.mean(loss_G_l1_test))
