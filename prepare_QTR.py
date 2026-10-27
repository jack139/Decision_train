import os
import json


out_path = 'data/QTR'

labels = {
    '3': "完全匹配",
    '2': "部分匹配",
    '1': "很少匹配",
    '0': "完全不匹配",
}

question = json.dumps({
    "relevance": {
        "type": "choice",
        "instructions": "评估搜索词(Query)表述主题和落地页标题(Title)表述主题的匹配程度。Query的主题是指query的专注点,用户在输入query是希望找到与query主题相关的网页。需要判断Query主题和Title主题是否一致及达到多大程度上的一致。",
        "criteria": {
            "完全匹配": "表示主题完全匹配",
            "部分匹配": "表示主题部分匹配",
            "很少匹配": "表示主题很少匹配，有一些参考价值",
            "完全不匹配": "表示主题完全不匹配或者没有参考价值",
        }
    }
}, ensure_ascii=False)


def convert_to_decision_dataset(filename):
    rows = []
    max_len = 0    

    for l in json.load(open(filename)):
        if len(l['query'])+len(l['title'])>128: # QTR
            max_len += 1

        row = {
            'id': l['id'],
            'questions': question,
            'state': json.dumps({"Query": l['query'], "Title": l['title']}, ensure_ascii=False),
            'gold': json.dumps({"relevance": labels[l['label']]}, ensure_ascii=False)
        }

        rows.append(row)

    _, fn = os.path.split(filename)

    json.dump(
        rows,
        open(os.path.join(out_path, fn+'_decision.json'), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    print(f"{fn}: {len(rows)} rows conveted.")


if __name__ == '__main__':
    filenames = [
        #'../../nlp/CBLUE/dataset/KUAKE-QTR/example_gold.json',
        '../../nlp/CBLUE/dataset/KUAKE-QTR/KUAKE-QTR_train.json',
        '../../nlp/CBLUE/dataset/KUAKE-QTR/KUAKE-QTR_dev.json',
    ]

    for f in filenames:
        convert_to_decision_dataset(f)
