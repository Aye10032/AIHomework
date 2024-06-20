import json
import random


def gen_general():
    result = []
    with open('testdata/vicuna-80-en.json', 'r', encoding='utf-8') as en_f:
        data_en = en_f.read()

    result.extend(json.loads(data_en))

    with open('testdata/vicuna-80-zh.json', 'r', encoding='utf-8') as zh_f:
        data_zh = zh_f.read()

    result.extend(json.loads(data_zh))

    general = random.choices(result, k=20)
    output_data = json.dumps(general)
    with open('general_test.json', 'w', encoding='utf-8') as f:
        f.write(output_data)


def main() -> None:
    gen_general()


if __name__ == '__main__':
    main()
