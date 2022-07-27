import os
import argparse
import numpy as np
import sys
import torch 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.dataset import dataset_gray
from net import net
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='3'

def main():

    model = net(in_channels=1, out_channels=1, n_feat=args.num_feat, kernel_size=3, stride=2, bias=False, n_classes=2)

    if args.gpu >= 0: model.cuda(args.gpu)
    print('| loading model file %s... ' % model_file)

    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    model.load_state_dict(pretrained_weight)
    print('done!')

    test_dataset  = dataset_gray(args.data_dir, args.split, have_label=None, transform=None)

    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader.n_iter = len(test_loader)
    model.eval()

    with torch.no_grad():
        for it, (images, images_ir, names) in enumerate(test_loader):
            images = Variable(images)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)

            fusion, _, _ = model(images, images_ir)

            print('name', names)

            save_fusion_path = './fusion_result/'
            os.makedirs(save_fusion_path,exist_ok=True)
            fusion = torch.clamp(fusion,0,1)
            fusion = fusion.cpu().numpy().squeeze(0).squeeze(0)
            cv2.imwrite(os.path.join(save_fusion_path,names[0]+'.png'),np.uint8(fusion*255))


        print('\n###########################################################################')
        print('\n| * the tested dataset name: %s' % args.dataset_name)
        print('| * the tested image count: %d' % test_loader.n_iter)
        print('| * the tested image size: %d*%d' %(args.img_height, args.img_width))
        print('\n###########################################################################')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test with pytorch')
    parser.add_argument('--pth_name',type=str,default='checkpoint.pth')
    parser.add_argument('--dataset_name', '-D', type=str, default='test')
    parser.add_argument('--img_height', '-IH', type=int, default=480) 
    parser.add_argument('--img_width', '-IW', type=int, default=640)  
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    parser.add_argument('--n_class', '-C',  type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='./dataset/')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_feat', type=int, default=32)
    parser.add_argument('--bilinear',dest='bilinear',action='store_true')
    args = parser.parse_args()

    batch_size = 1
    torch.cuda.set_device(args.gpu)
    print("\n| the gpu count:", torch.cuda.device_count())
    print("| the current used gpu:", torch.cuda.current_device(), '\n')

    model_dir = args.model_dir
    if os.path.exists(model_dir) is False:
        print("| the %s does not exit." %(model_dir))
        sys.exit()
    model_file = os.path.join(model_dir, args.pth_name)
    if os.path.exists(model_file) is True:
        print('| use the final model file.')
    else:
        print('| no model file found.')
        sys.exit()
    main()
