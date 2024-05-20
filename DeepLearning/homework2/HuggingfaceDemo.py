import evaluate
import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer

ds = load_dataset('cifar10')
metric = evaluate.load("accuracy")

labels = ds['train'].features['label'].names

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['img']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def main() -> None:
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    prepared_ds = ds.with_transform(transform)

    training_args = TrainingArguments(
        output_dir="./vit-base-cifar10",
        per_device_train_batch_size=32,
        eval_strategy="steps",
        num_train_epochs=5,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    outputs = trainer.predict(prepared_ds['test'])
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    _labels = prepared_ds['test'].features['label'].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=_labels)
    disp.plot(xticks_rotation=45)
    plt.savefig('ConfusionMatrix.png')


if __name__ == '__main__':
    main()
