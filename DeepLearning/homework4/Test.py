import argparse
from dataclasses import asdict

import evaluate
import torch
from accelerate import load_checkpoint_in_model
from evaluate import EvaluationModule
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Config import ModelConfig, BOS_IDX, EOS_IDX
from Data import TransData, DataType, Tokenizer
from Model import Transformer


def translate(net: Transformer, input_token: Tensor, max_length: int) -> Tensor:
    encoder_out = net.encode(input_token)

    output_token = [BOS_IDX]
    while len(output_token) < max_length:
        decoder_in = torch.LongTensor(output_token).unsqueeze(0).cuda()
        decoder_out = net.decode(decoder_in, input_token, encoder_out)

        next_word_probs: Tensor = decoder_out[0, decoder_out.shape[1] - 1, :]
        next_word_id = torch.argmax(next_word_probs)
        output_token.append(next_word_id.item())

        if next_word_id == EOS_IDX:
            break

    return torch.LongTensor(output_token)


def valid(net: Transformer, dataloader: DataLoader, tokenizer: Tokenizer, metric: EvaluationModule):
    for index, (src, target) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f'Valid '
    ):
        src: Tensor = src.cuda()
        target: Tensor = target.cuda()
        real_target = target[:, 1:]
        input_target = target[:, :-1]

        outputs: Tensor = net(src, input_target)

        real_sentence = tokenizer.detokenize(real_target.squeeze())
        output_sentence = tokenizer.detokenize(torch.argmax(outputs, -1).squeeze())
        metric.add_batch(predictions=[output_sentence], references=[real_sentence])

    result = metric.compute()
    logger.info(round(result['bleu'], 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--bleu', action='store_true', default=False)
    args = parser.parse_args()

    # dataloader_config = DataLoaderConfiguration(split_batches=True)
    # accelerator = Accelerator(dataloader_config=dataloader_config)

    dataset = TransData('data', DataType.TEST, True)

    config = ModelConfig.default_config()
    net = Transformer(**asdict(config))
    load_checkpoint_in_model(net, 'model')
    net = net.cuda()
    net.eval()

    if args.bleu:
        bleu_metric = evaluate.load('bleu')

        dataloader = DataLoader(dataset, batch_size=1, num_workers=2)
        valid(net, dataloader, dataset.tgt_tokenizer, bleu_metric)

    if args.input != '':
        assert len(args.input) <= config.max_token

        sentences = ['<BOS>'] + args.input.strip().split(' ') + ['<EOS>']
        input_token = dataset.src_tokenizer.tokenize(sentences).unsqueeze(0).cuda()
        output_token = translate(net, input_token, config.max_token)
        print(dataset.tgt_tokenizer.detokenize(output_token))


if __name__ == '__main__':
    main()
