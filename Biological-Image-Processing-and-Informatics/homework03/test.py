import argparse
import os
import random
from glob import glob

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
# from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

from dataset import Dataset
from model import UNet
from utils import AverageMeter
from utils import dice_coef


def set_seed(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='UNet',
                        help='model name')
    parser.add_argument('--dataset', default='ER',
                        help='dataset name, ER or MITO')
    parser.add_argument('--loss', default='BCELoss',
                        help='loss define')
    parser.add_argument('--optimizer', default='SGD', help='optimizer define')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    exp_dir = os.path.join(args.name, f"{args.dataset}-{args.loss}-{args.optimizer}-lr{args.lr}")

    with open(f'{exp_dir}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['name'])
    model = UNet()
    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('data', config['dataset'] + '_dataset', 'test', 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    model.load_state_dict(torch.load(f'{exp_dir}/model.pth'))
    model.eval()

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
    ])

    test_dataset = Dataset(
        img_ids=img_ids,
        img_dir=os.path.join('data', config['dataset'] + '_dataset', 'test', 'images'),
        mask_dir=os.path.join('data', config['dataset'] + '_dataset', 'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    os.makedirs(os.path.join('outputs', config['name']), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)

            dice = dice_coef(output > 0.5, target)
            avg_meter.update(dice, input.size(0))

            # output = output.squeeze().cpu().numpy()
            # for i in range(len(output)):
            #     cv2.imwrite(os.path.join(exp_dir, meta['img_id'][i] + '.tif'),
            #                 (output[i] * 255).astype('uint8'))

    print('Dice: %.4f' % avg_meter.avg)

    plot_examples(input, target, model, exp_dir, num_examples=3)

    torch.cuda.empty_cache()


def plot_examples(datax, datay, model, exp_dir, num_examples=6):
    set_seed()
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx + 1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0, :, :].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][2].set_title("Target image")
    plt.tight_layout()
    # save the plot
    plt.savefig(os.path.join(exp_dir, 'examples.png'))


if __name__ == '__main__':
    main()
