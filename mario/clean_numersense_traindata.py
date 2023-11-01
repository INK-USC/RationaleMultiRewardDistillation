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
    # getting data into question-indexed dictionary
    data = {"train":{}, "test":{}}
    classes = {"no": "(A)", "zero": "(B)", "one": "(C)", "two": "(D)", "three": "(E)",\
                "four": "(F)", "five": "(G)", "six": "(H)", "seven": "(I)", "eight": "(J)",\
                "nine": "(K)", "ten": "(L)"}
    # original train
    d = pd.read_csv('data/numersense/numersense.train.masked.tsv', delimiter='\t', header=None)
    d.columns = ["question", "answer"]
    for i in range(len(d)):
        data["train"][d['question'].iloc[i] + " \n (A) no (B) zero (C) one (D) two (E) three (F) four (G) five (H) six (I) seven (J) eight (K) nine (L) ten"] = classes[d['answer'].iloc[i]]
    del(d)
    # test
    d = pd.read_csv('data/numersense/numersense.validation.masked.tsv', delimiter='\t', header=None)
    d.columns = ["question", "answer"]
    for i in range(len(d)):
        data["test"][d['question'].iloc[i] + " \n (A) no (B) zero (C) one (D) two (E) three (F) four (G) five (H) six (I) seven (J) eight (K) nine (L) ten"] = classes[d['answer'].iloc[i]]

    # silver-rationale data from GPT-3
    train_loaded_w_silver_rationales = pd.read_json(
        'numersense/gpt3/train_gpt3_responses.jsonl',
        orient='records')
    train_loaded_w_silver_rationales = train_loaded_w_silver_rationales.T

    # raw train data to be saved
    running_idx = 0
    dev_running_idx = 0
    train_data_jsonl = {}
    dev_data_jsonl = {}
    f1=open("data/numersense/raw/train.tsv", "w")
    f1.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    f2=open("data/numersense/raw/dev.tsv", "w")
    f2.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")

    # saving some dev questions
    dev_inds = sorted(list(np.random.choice(np.arange(0,len(data["train"])), 500, replace=False)))
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
    with open("data/numersense/raw/train.jsonl", "w") as f:
        json.dump(train_data_jsonl, f)
    with open("data/numersense/raw/dev.jsonl", "w") as f:
        json.dump(dev_data_jsonl, f)

    for q in dev_questions1:
        assert q not in train_questions1
    del(dev_data_jsonl)
    del(train_data_jsonl)

    # raw test data to be saved # original val
    f=open("data/numersense/raw/test.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    test_data_jsonl = {}
    for qq in data["test"]:
        ct_question = qq
        rationale = "dummy"
        gold_choice = data["test"][qq]

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        test_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/numersense/raw/test.jsonl", "w") as f:
        json.dump(test_data_jsonl, f)


if __name__=="__main__":
    main()
