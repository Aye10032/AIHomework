import json
import os
import random

import pandas as pd


def about_me():
    instructions = ['talk about yourself', '你是谁？', '介绍一下你自己', 'who are you?', 'what`s your name?', '你是谁？']
    messages = [{
        'instruction': instruction,
        'input': '',
        'output': '你好！我是由第8小组开发的一个具有10个神经元的单层感知机网络，你可以向我提出你的问题。'
    } for instruction in instructions]

    return messages


def clean_csv():
    datas = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('csv'):
                data = pd.read_csv(os.path.join(root, file), encoding='utf-8')
                old = data.shape[0]
                data = data.dropna()
                print(f'{old} -> {data.shape[0]}')
                datas.append(data)

    final = pd.concat(datas)
    final.to_csv('gene.csv', index=False, encoding='utf-8')


def concat_json(about: list):
    result = []

    df = pd.read_csv('gene.csv', encoding='utf-8')
    for index, row in df.iterrows():
        gene_name = row.gene
        summary: str = row.summary
        location = row.location

        instructions = [
            f'Talk about gene {gene_name}.',
            f'简要描述一下基因{gene_name}',
            f'What do you know about gene {gene_name}?',
            f'关于基因{gene_name}你知道些什么？',
            f'Discuss the functions of {gene_name} in genetic processes.',
            f'讨论{gene_name}在遗传过程中的功能',
            f'Explain the role of {gene_name} in the development of an organism',
            f'解释{gene_name}在生物体发育过程中的作用',
            f'Elaborate on the significance of {gene_name} in cellular functions',
            f'阐述{gene_name}在细胞功能中的意义'
        ]
        messages = [{
            'instruction': instruction,
            'input': '',
            'output': summary.replace('The gene', gene_name).replace('This gene', gene_name)
        } for instruction in instructions]
        result.extend(messages)

        instructions = [
            f'Where is the gene {gene_name} located?',
            f'基因{gene_name}位于哪里？',
            f'What is the chromosomal location of the {gene_name} gene?',
            f'{gene_name}基因在染色体上的位置是什么？'
        ]
        messages = [{
            'instruction': instruction,
            'input': '',
            'output': location
        } for instruction in instructions]
        messages = messages * 2
        result.extend(messages)

    json_data = json.dumps(result)
    with open('task_data.json', 'w', encoding='utf-8') as f:
        f.write(json_data)

    print(len(result))

    about = about * 500
    print(len(about))
    result.extend(about)

    with open('alpaca_data.json', 'r', encoding='utf-8') as f:
        json_data = f.read()
    origin_data = json.loads(json_data)
    result.extend(origin_data)

    random.shuffle(result)

    json_data = json.dumps(result)
    with open('self_data.json', 'w', encoding='utf-8') as f:
        f.write(json_data)


def main() -> None:
    about = about_me()

    clean_csv()
    concat_json(about)

    with open('self_data.json', 'r', encoding='utf-8') as f:
        read_data = f.read()

    json_data = json.loads(read_data)
    print(len(json_data))


if __name__ == '__main__':
    main()
