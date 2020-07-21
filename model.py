import os
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils import normalize, denormalize, set_requires_grad
from pix2pixHD import pix2pixStyle, Discriminator_p, Discriminator_t
from loss import criterion_GAN, criterionPerPixel


class pix2pixHDModel:
    def __init__(self, opt, trainloader, testloader):
        self.h, self.w, self.c = opt.img_size
        self.inputc = opt.inputc
        self.posec = opt.posec
        self.dt_inputc = opt.dt_inputc
        self.dp_inputc = opt.dp_inputc

        self.lr = opt.lr

        self.epochs = opt.epochs
        self.save_dir = opt.save_dir
        self.load_dir = opt.load_dir
        self.show_per_epoch = opt.show_per_epoch
        self.test_per_epoch = opt.test_per_epoch
        self.save_per_epoch = opt.save_per_epoch
        self.use_lsgan = opt.use_lsgan
        self.use_sigmoid = opt.use_sigmoid

        self.trainloader = trainloader
        self.testloader = testloader

        self.pix2pix = pix2pixStyle(self.inputc, norm_layer=nn.BatchNorm2d)
        self.dnet_t = Discriminator_t(self.dt_inputc, norm_layer=nn.BatchNorm2d, use_sigmoid=self.use_sigmoid)
        self.dnet_p = Discriminator_p(self.dp_inputc, norm_layer=nn.BatchNorm2d, use_sigmoid=self.use_sigmoid)

        self.criterionGAN = criterion_GAN(use_lsgan=self.use_lsgan) 
        self.criterionPerPixel = criterionPerPixel()
        self.criterionFM = nn.L1Loss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    reference:
        https://github.com/vt-vl-lab/Guided-pix2pix
    """
    def init(self, init_type='normal', init_gain=0.02):
        self.init_weights(self.pix2pix, init_type, gain=init_gain)
        self.init_weights(self.dnet_t, init_type, gain=init_gain)
        self.init_weights(self.dnet_p, init_type, gain=init_gain)
        print('initialize network with %s' % init_type)


    def init_weights(self, net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
        net.apply(init_func)

    def save_model(self):
        torch.save(self.pix2pix.state_dict(), os.path.join(self.save_dir, 'g_net.pth'))
        torch.save(self.dnet_t.state_dict(), os.path.join(self.save_dir, 'dnet_t.pth'))
        torch.save(self.dnet_p.state_dict(), os.path.join(self.save_dir, 'dnet_p.pth'))


    def load_model(self, load_g = True, load_d = True):
        if load_g:
            gnet_dict = torch.load(os.path.join(self.load_dir, 'g_net.pth'))
            self.pix2pix.load_state_dict(gnet_dict)
        if load_d:
            dt_dict = torch.load(os.path.join(self.load_dir, 'dnet_t.pth'))
            self.dnet_t.load_state_dict(dt_dict)
            dp_dict = torch.load(os.path.join(self.load_dir, 'dnet_p.pth'))
            self.dnet_p.load_state_dict(dp_dict)

    def train(self):
        print(f'Use device: {self.device}'
              f'Network:\n'
              f'In channels: {self.c}\n'
              f'Out channels: 3')
              
        self.pix2pix.to(device=self.device)
        self.dnet_t.to(device=self.device)
        self.dnet_p.to(device=self.device)


        self.optimizer_G = torch.optim.Adam(self.pix2pix.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_DT = torch.optim.Adam(self.dnet_t.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_DP = torch.optim.Adam(self.dnet_p.parameters(), lr=self.lr, betas=(0.5, 0.999))

        for epoch in range(self.epochs):
            err_g_list = []
            loss_l1_list = []
            loss_p_list = []
            loss_sty_list = []
            lg_t_list = []
            lg_p_list = []
            loss_dt_list = []
            loss_dp_list = []


            for idx, data_batch in enumerate(self.trainloader):
                for key in data_batch['input'].keys():
                    data_batch['input'][key] = normalize(data_batch['input'][key])

                img = data_batch['input']['image']
                t_pose = data_batch['input']['t_pose']
                target = data_batch['input']['target']
                style = data_batch['input']['n_img']
                # smap = data_batch['input']['s_map']
                # t_map = data_batch['input']['t_map']

                self.img = img.type(torch.FloatTensor).to(self.device)
                self.t_pose = t_pose.type(torch.FloatTensor).to(self.device)
                self.target = target.type(torch.FloatTensor).to(self.device)
                self.style = style.type(torch.FloatTensor).to(self.device)
                # self.smap = smap.type(torch.FloatTensor).to(self.device)
                # self.t_map = t_map.type(torch.FloatTensor).to(self.device)

                err_g, err_dt, err_dp, loss_l1, loss_p, loss_style, l_gan_t, l_gan_p = self.one_step()

                err_g_list.append(err_g.item())
                loss_l1_list.append(loss_l1.item())
                loss_p_list.append(loss_p.item())
                loss_sty_list.append(loss_style.item())
                lg_t_list.append(l_gan_t.item())
                lg_p_list.append(l_gan_p.item())
                loss_dt_list.append(err_dt.item())
                loss_dp_list.append(err_dp.item())

                if idx % self.show_per_epoch == 0:
                    print('Epoch: {}, Iter: {}, gloss: {:.5f}, l1: {:.5f}, '
                          'lp: {:.5f}, l_style: {:.5f}, lg_t: {:.5f}, lg_p: {:.5f}, ld_t: {:.5f}, ld_p: {:.5f}'.format(
                           epoch, idx, np.mean(err_g_list), np.mean(loss_l1_list),np.mean(loss_p_list), 
                           np.mean(loss_sty_list), np.mean(lg_t_list), np.mean(lg_p_list), 
                           np.mean(loss_dt_list), np.mean(loss_dp_list)))

                    err_g_list.clear()
                    loss_l1_list.clear()
                    loss_p_list.clear()
                    loss_sty_list.clear()
                    lg_t_list.clear()
                    lg_p_list.clear()
                    loss_dt_list.clear()
                    loss_dp_list.clear()

                if idx % self.test_per_epoch == 0:
                    self.test(epoch, idx)
                if idx % self.save_per_epoch == 0:
                    self.save_model()

    def one_step(self):
        self.fake = self.pix2pix(self.img, self.t_pose, self.style)

        # dicscriminator_t forward
        real_score_dt = self.dnet_t(torch.cat((self.target, self.t_pose), dim=1))
        real_loss_dt = self.criterionGAN(real_score_dt, True)

        fake_score_dt = self.dnet_t(torch.cat((self.fake.detach(), self.t_pose), dim=1))
        fake_loss_dt = self.criterionGAN(fake_score_dt, False)

        err_dt = (real_loss_dt + fake_loss_dt) * 0.5

        # dicscriminator_p forward
        real_score_dp = self.dnet_p(torch.cat((self.target, self.img), dim=1))
        real_loss_dp = self.criterionGAN(real_score_dp, True)

        fake_score_dp = self.dnet_p(torch.cat((self.fake.detach(), self.img), dim=1))
        fake_loss_dp = self.criterionGAN(fake_score_dp, False)

        err_dp = (real_loss_dp + fake_loss_dp) * 0.5

        # generator forward
        fake_score_gt = self.dnet_t(torch.cat((self.fake, self.t_pose), dim=1))
        l_gan_t = self.criterionGAN(fake_score_gt, True)

        fake_score_gp = self.dnet_p(torch.cat((self.fake, self.img), dim=1))
        l_gan_p = self.criterionGAN(fake_score_gp, True)

        # l_fm = 0
        # for i in range(len(real_inter_d)):
        #     for j in range(len(real_inter_d[i])):
        #         l_fm += self.criterionFM(real_inter_d[i][j].detach(), fake_inter_g[i][j])

        loss_l1, loss_p, loss_style = self.criterionPerPixel(self.fake, self.target)
        err_g = 1 * loss_l1 + 2 * loss_p + 0 * loss_style + 0.5 * l_gan_t + 0.5 * l_gan_p

        # optimize genrator
        self.pix2pix.zero_grad()
        err_g.backward()
        self.optimizer_G.step()

        # optimize discriminator_t
        self.dnet_t.zero_grad()
        err_dt.backward()
        self.optimizer_DT.step()

        # optimize discriminator_p
        self.dnet_p.zero_grad()
        err_dp.backward()
        self.optimizer_DP.step()

        return err_g, err_dt, err_dp, loss_l1, loss_p, loss_style, l_gan_t, l_gan_p

    def test(self, epoch=0, step=0, save_pic=True):
        # load test sample from testloader
        test_batch = iter(self.testloader).next()
        for key in test_batch['input'].keys():
            test_batch['input'][key] = normalize(test_batch['input'][key])
        test_img = test_batch['input']['image']
        test_img = test_img.type(torch.FloatTensor).to(self.device)
        test_pose = test_batch['input']['t_pose']
        test_pose = test_pose.type(torch.FloatTensor).to(self.device)
        test_style = test_batch['input']['n_img']
        test_style = test_style.type(torch.FloatTensor).to(self.device)
        # test_map = test_batch['input']['s_map']
        # test_map = test_map.type(torch.FloatTensor).to(self.device)
        # test_t_map = test_batch['input']['t_map']
        # test_t_map = test_t_map.type(torch.FloatTensor).to(self.device)

        with torch.no_grad():
            source = (test_batch['input']['image']).detach().cpu()
            poses = (test_batch['input']['t_pose']).detach().cpu()
            poses = denormalize(poses)
            poses = torch.unsqueeze(torch.sum(poses, dim=1), dim=1)
            target_ = (test_batch['input']['target']).detach().cpu()

            fake_ = self.pix2pix(test_img, test_pose, test_style)
            fake_ = fake_.detach().cpu()
        
        fake_ = denormalize(fake_)
        target_ = denormalize(target_)
        source = denormalize(source)

        sample_source = vutils.make_grid(source, padding=2, normalize=False)
        sample_pose = vutils.make_grid(poses, padding=2, normalize=False)
        sample_target = vutils.make_grid(target_, padding=2, normalize=False)
        sample_fake = vutils.make_grid(fake_, padding=2, normalize=False)
        plt.figure(figsize=(32, 16))
        plt.axis('off')
        plt.title('fake image')
        plt.subplot(4, 1, 1)
        plt.imshow(np.transpose(sample_source, (1, 2, 0)))
        plt.subplot(4, 1, 2)
        plt.imshow(np.transpose(sample_pose, (1, 2, 0)))
        plt.subplot(4, 1, 3)
        plt.imshow(np.transpose(sample_fake, (1, 2, 0)))
        plt.subplot(4, 1, 4)
        plt.imshow(np.transpose(sample_target, (1, 2, 0)))
        if save_pic:
            plt.savefig("./output/epoch_{}_iter_{}.png".format(epoch, step))
        plt.close()