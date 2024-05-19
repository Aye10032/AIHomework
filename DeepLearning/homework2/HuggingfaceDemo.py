import os.path
import shutil

import torch
from torchvision.transforms import (
    Compose,
    Resize,
    RandomResizedCrop,
    CenterCrop,
    RandomHorizontalFlip,
    RandomRotation,
    ToTensor,
    Normalize,
    ToPILImage
)
from torchvision.datasets import CIFAR10
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

# def load_data():
if os.path.exists('./cifar10'):
    shutil.rmtree('./cifar10')

train_data = load_dataset('cifar10', cache_dir='./cifar10', split='train')
test_set = load_dataset('cifar10', cache_dir='./cifar10', split='test')
splits = train_data.train_test_split(test_size=0.1)
train_set = splits['train']
valid_set = splits['test']

i_to_s = dict((k, v) for k, v in enumerate(train_set.features['label'].names))
s_to_i = dict((v, k) for k, v in enumerate(train_set.features['label'].names))

set_trans = Compose([
    Resize(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def trans(args):
    args['pixels'] = [set_trans(image.convert('RGB')) for image in args['img']]
    return args


train_set.set_transform(trans)
valid_set.set_transform(trans)
test_set.set_transform(trans)

processor: ViTImageProcessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model: ViTForImageClassification = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=10,
    ignore_mismatched_sizes=True,
    id2label=i_to_s,
    label2id=s_to_i
)
