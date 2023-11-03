# cleaning the training data for OBQA, 5 rationales are sampled from GPT-3 (text-davinci-003)
# rationales have to be filtered based on whether they predict the correct answer or not
import json
from collections import defaultdict
from datasets import load_dataset
import pandas as pd

def reorganize_pools(pools):
    by_index = defaultdict(list)
    for example in pools:
        if example["score"]==1:
            by_index[example["index"]].append(example)
    return by_index

def main():
    d=load_dataset("quarel", "main")

    # getting data from HF into question-indexed dictionary
    data = {"train":{}, "validation":{}, "test":{}}
    for split in ["train", "validation", "test"]:
        for dd in d[split]:
            ct_question = dd["question"]
            data[split][ct_question] = dd
    # answer dictionary
    answer_dict = {0: "(A)", 1: "(B)"}

    # silver-rationale data from GPT-3
    train_loaded_w_silver_rationales = pd.read_json(
        'quarel/gpt3/train_gpt3_responses.jsonl',
        orient='records')
    train_loaded_w_silver_rationales = train_loaded_w_silver_rationales.T

    # raw train data to be saved
    running_idx = 0
    train_data_jsonl = {}
    f=open("data/quarel/raw/train.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    for i in range(len(train_loaded_w_silver_rationales)):
        ct_question = train_loaded_w_silver_rationales.iloc[i]['question']
        ct_gold_label = train_loaded_w_silver_rationales.iloc[i]['gold_label'].upper()
        ct_pred_label = train_loaded_w_silver_rationales.iloc[i]['predicted_label'].upper()
        ct_pred_rationale = train_loaded_w_silver_rationales.iloc[i]['predicted_rationale']

        if len(ct_pred_rationale) == 0:
            continue
        if ct_pred_rationale[-1] != '.':
            ct_pred_rationale = ct_pred_rationale + '.'
        if ct_gold_label != ct_pred_label:
            continue
        
        f.write(ct_question + "\t" + ct_pred_rationale + "\t" + ct_pred_label + "\n")
        train_data_jsonl[running_idx] = {'question': ct_question, 'rationale': ct_pred_rationale, 'answer': ct_pred_label}
        running_idx += 1
    f.close()
    with open("data/quarel/raw/train.jsonl", "w") as f:
        json.dump(train_data_jsonl, f)

    # raw dev data to be saved
    f=open("data/quarel/raw/dev.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    dev_data_jsonl = {}
    for qq in data["validation"]:
        ct_question = qq
        rationale = "dummy"
        gold_choice = data["validation"][ct_question]["answer_index"]
        gold_choice = answer_dict[gold_choice]

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        dev_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/quarel/raw/dev.jsonl", "w") as f:
        json.dump(dev_data_jsonl, f)

    # raw test data to be saved
    f=open("data/quarel/raw/test.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    test_data_jsonl = {}
    for qq in data["test"]:
        ct_question = qq
        rationale = "dummy"
        gold_choice = data["test"][ct_question]["answer_index"]
        gold_choice = answer_dict[gold_choice]

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        test_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/quarel/raw/test.jsonl", "w") as f:
        json.dump(test_data_jsonl, f)

if __name__=="__main__":
    main()
