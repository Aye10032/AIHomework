import argparse

import torch

from Data import TangData
from main import PoetryModel


def generate(model, start_words, ix2word, word2ix):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    inputs = torch.tensor([word2ix['<START>']], device='cuda').view(1, 1).long()

    hidden = None
    model.eval()
    with torch.no_grad():
        for i in range(50):
            output, hidden = model(inputs, hidden)
            # 如果在给定的句首中，input 为句首中的下一个字
            if i < start_words_len:
                w = results[i]
                inputs = inputs.data.new([word2ix[w]]).view(1, 1)
            # 否则将 output 作为下一个 input 进行
            else:
                top_index = output.data[0].topk(1)[1][0].item()
                w = ix2word[top_index]
                results.append(w)
                inputs = inputs.data.new([top_index]).view(1, 1)
            if w == '<EOP>':
                del results[-1]
                break

    return ''.join(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='')
    args = parser.parse_args()

    dataset = TangData()
    ix2word = dataset.ix2word
    word2ix = dataset.word2ix

    model = PoetryModel(len(word2ix), 128, 1024, 3).cuda()
    model.load_state_dict(torch.load('model/model.pth'))

    model_output = generate(model, args.input, ix2word, word2ix)
    print(model_output)


if __name__ == '__main__':
    main()
