import os, argparse, time
import SSIM
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util.dataset import dataset_gray
from util.augmentation import RandomCrop2_three, RandomFlip_three
from tensorboardX import SummaryWriter
from net import net
import wse

def train(epo, model, train_loader, optimizer):
    single_ssim_loss = SSIM.ssim
    WSE = wse.WSELoss
    if args.lr_reduce == 1:
        lr_this_epo = args.lr_start * args.lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
    else:
        if epo >= 0 and epo < 40:
            lr_this_epo = 1e-3
        elif epo >= 40 and epo < 80:
            lr_this_epo = 1e-4
        elif epo >= 80 and epo < 100:
            lr_this_epo = 1e-5
        elif epo >= 100 and epo <= 120:
            lr_this_epo = 1e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

    start_t = time.time()
    model.train()

    for it, (images, images_ir, names) in enumerate(train_loader):

        images = Variable(images, requires_grad=True).cuda()
        images_ir = Variable(images_ir, requires_grad=True).cuda()
        if torch.cuda.is_available():
            images = images.cuda()
            images_ir = images_ir.cuda()

        optimizer.zero_grad()

        fusion, ir_degradation, vis_degradation = model(images, images_ir)

        loss_fusion = 1 * WSE()(fusion, images, fusion, images_ir)
        loss_ssim = (1. - single_ssim_loss(fusion, images)) + (1. - single_ssim_loss(fusion, images_ir))
        loss_ssim_degradation = (1. - 1 * single_ssim_loss(vis_degradation, images)) + (1. - 1 * single_ssim_loss(ir_degradation, images_ir))
        loss_degradation = 1 * WSE()(ir_degradation, images_ir, vis_degradation, images)

        loss = loss_fusion + 10 * loss_ssim + (0.2 * loss_degradation + 2 * loss_ssim_degradation)

        loss.backward()
        optimizer.step()

        print('|- epo %s/%s, lr %s, train iter %s/%s, %.2f img/sec loss: %.4f, loss_fusion: %.4f, loss_ssim: %.8f' \
              % ( epo, args.epoch_max, lr_this_epo, it + 1, train_loader.n_iter,
                 (it + 1) * args.batch_size / (time.time() - start_t), float(loss), float(loss_fusion),
                 float(loss_degradation)))

        writer.add_scalar('loss', loss, epo)


def set_optimizer_scheduler(args, model, epoch):
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr_start,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epoch, eta_min=0.0)
    return optimizer, scheduler


def main():
    data_dir = './dataset/'

    augmentation_methods = [
        RandomFlip_three(prob=0.5),
        RandomCrop2_three(crop_h=args.crop_size, crop_w=args.crop_size, prob=1.0),
    ]

    train_dataset = dataset_gray(data_dir, 'train', have_label=True, transform=augmentation_methods)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    train_loader.n_iter = len(train_loader)

    for epo in range(args.epoch_from, args.epoch_max):
        print('\nepo #%s begin...' % (epo))

        train(epo, model, train_loader, optimizer)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('|- saving check point %s: ' % checkpoint_model_file)
        torch.save(model.module.state_dict(), checkpoint_model_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train with pytorch')
    #############################################################################################
    parser.add_argument('--save_model_name', '-SN', type=str, default='checkpoint')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--lr_start', '-LS', type=float, default=0.001)
    parser.add_argument('--beta', '-BE', type=float, default=10)
    parser.add_argument('--gpu', '-G', type=str, default=0)
    parser.add_argument('--optim', '-O', type=str, default=0)
    #############################################################################################
    parser.add_argument('--lr_decay', '-LD', type=float, default=0.99)
    parser.add_argument('--epoch_max', '-E', type=int, default=120)
    parser.add_argument('--epoch_from', '-EF', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--crop_size', '-C', type=int, default=256)
    parser.add_argument('--num_feat', '-N', type=int, default=32)
    parser.add_argument('--lr_reduce', '-R', type=int, default=0)
    parser.add_argument('--bilinear', dest='bilinear', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    args = parser.parse_args()

    n_class = 2
    print("\nthe gpu count:", torch.cuda.device_count())


    model = net(in_channels=1, out_channels=1, n_feat=args.num_feat, kernel_size=3, stride=2, bias=False, n_classes=2)

    model = model.cuda()
    print(model)
    model = torch.nn.DataParallel(model)
    scheduler = None
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    elif args.optim == 'sgd_cos':
        optimizer, scheduler = set_optimizer_scheduler(args, model, args.epoch_max)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0.0005, amsgrad=False)

    weight_dir = os.path.join('./model/', args.save_model_name)
    os.makedirs(weight_dir, exist_ok=True)

    tensorboard_log = os.path.join('train_log/', args.save_model_name)
    os.makedirs(tensorboard_log, exist_ok=True)

    # tensorboardX setup
    writer = SummaryWriter(tensorboard_log)

    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    main()
