# cleaning the training data for OBQA, 5 rationales are sampled from GPT-3 (text-davinci-003)
# rationales have to be filtered based on whether they predict the correct answer or not
import json
from collections import defaultdict
from datasets import load_dataset
import pandas as pd
import numpy as np

def reorganize_pools(pools):
    by_index = defaultdict(list)
    for example in pools:
        if example["score"]==1:
            by_index[example["index"]].append(example)
    return by_index

def main():
    d=load_dataset("qasc", "main")

    # getting data from HF into question-indexed dictionary
    data = {"train":{}, "validation":{}, "test":{}}
    for split in ["train", "validation", "test"]:
        for dd in d[split]:
            ct_question = dd["formatted_question"]
            data[split][ct_question] = dd
    # answer dictionary
    #answer_dict = {0: "(A)", 1: "(B)"}

    # silver-rationale data from GPT-3
    train_loaded_w_silver_rationales = pd.read_json(
        'qasc/gpt3/train_gpt3_responses.jsonl',
        orient='records')
    train_loaded_w_silver_rationales = train_loaded_w_silver_rationales.T

    # raw train data to be saved
    running_idx = 0
    dev_running_idx = 0
    train_data_jsonl = {}
    dev_data_jsonl = {}
    f1=open("data/qasc/raw/train.tsv", "w")
    f1.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    f2=open("data/qasc/raw/dev.tsv", "w")
    f2.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")

    # saving some dev questions
    dev_inds = sorted(list(np.random.choice(np.arange(0,len(data["train"])), 900, replace=False)))
    dev_questions = []
    i = 0
    for qq in data["train"]:
        if i in dev_inds:
            dev_questions.append(qq)
        i += 1
    print("len dev:", len(list(set(dev_questions))))

    dev_questions1 = []
    train_questions1 = []
    for i in range(len(train_loaded_w_silver_rationales)):
        ct_question = train_loaded_w_silver_rationales.iloc[i]['question']
        ct_gold_label = train_loaded_w_silver_rationales.iloc[i]['gold_label'].upper()
        ct_pred_label = train_loaded_w_silver_rationales.iloc[i]['predicted_label'].upper()
        ct_pred_rationale = train_loaded_w_silver_rationales.iloc[i]['predicted_rationale']

        if len(ct_pred_rationale) == 0:
            continue
        if ct_pred_rationale[-1] != '.':
            ct_pred_rationale = ct_pred_rationale + '.'

        if ct_question in dev_questions:
            if ct_question not in dev_questions1: # 2nd condition is to prevent rpeeats in tsv
                f2.write(ct_question + "\t" + "dummy" + "\t" + ct_gold_label + "\n")
                dev_data_jsonl[dev_running_idx] = {'question': ct_question, 'rationale': "dummy", 'answer': ct_gold_label}
                dev_questions1.append(ct_question)
                dev_running_idx += 1
            continue


        if ct_gold_label != ct_pred_label:
            continue
        
        f1.write(ct_question + "\t" + ct_pred_rationale + "\t" + ct_pred_label + "\n")
        train_data_jsonl[running_idx] = {'question': ct_question, 'rationale': ct_pred_rationale, 'answer': ct_pred_label}
        train_questions1.append(ct_question)
        running_idx += 1
    f1.close()
    f2.close()
    with open("data/qasc/raw/train.jsonl", "w") as f:
        json.dump(train_data_jsonl, f)
    with open("data/qasc/raw/dev.jsonl", "w") as f:
        json.dump(dev_data_jsonl, f)

    for q in dev_questions1:
        assert q not in train_questions1
    del(dev_data_jsonl)
    del(train_data_jsonl)

    # raw test data to be saved # original val
    f=open("data/qasc/raw/test.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    test_data_jsonl = {}
    for qq in data["validation"]:
        ct_question = qq
        dd = data["validation"][ct_question]
        rationale = "dummy"
        gold_choice = '(' + dd['answerKey'] + ')'
        #gold_choice = data["validation"][ct_question]["answer_index"]
        #gold_choice = answer_dict[gold_choice]

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        test_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/qasc/raw/test.jsonl", "w") as f:
        json.dump(test_data_jsonl, f)


if __name__=="__main__":
    main()
