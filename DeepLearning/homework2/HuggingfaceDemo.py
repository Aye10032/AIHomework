import os.path
import shutil

import numpy as np
import torch
from accelerate import init_empty_weights
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch import Tensor
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
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# def load_data():
# if os.path.exists('./cifar10'):
#     shutil.rmtree('./cifar10')

train_data = load_dataset('cifar10', split='train')
test_set = load_dataset('cifar10', split='test')
splits = train_data.train_test_split(test_size=0.1)
train_set = splits['train']
valid_set = splits['test']

i_to_s = dict((k, v) for k, v in enumerate(train_set.features['label'].names))
s_to_i = dict((v, k) for k, v in enumerate(train_set.features['label'].names))


def train_trans(args):
    _trans = Compose([
        Resize(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    args['pixels'] = [_trans(image.convert('RGB')) for image in args['img']]
    return args


def valid_trans(args):
    _trans = Compose([
        Resize(224),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    args['pixels'] = [_trans(image.convert('RGB')) for image in args['img']]
    return args


train_set.set_transform(train_trans)
valid_set.set_transform(valid_trans)
test_set.set_transform(valid_trans)

processor: ViTImageProcessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

with init_empty_weights():
    model: ViTForImageClassification = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=10,
        ignore_mismatched_sizes=True,
        id2label=i_to_s,
        label2id=s_to_i
    )

model.tie_weights()

train_args = TrainingArguments(
    'test-cifar10',
    save_strategy='epoch',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_dir='hf_log',
    report_to=['tensorboard'],
    remove_unused_columns=False
)


def collate_fn(examples):
    pixels = torch.stack([example["pixels"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixels, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


trainer = Trainer(
    model,
    train_args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()

outputs = trainer.predict(test_set)
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

_labels = train_set.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=_labels)
disp.plot(xticks_rotation=45)
plt.savefig('ConfusionMatrix.png')
