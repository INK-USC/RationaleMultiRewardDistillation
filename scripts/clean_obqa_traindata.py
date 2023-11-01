# cleaning the training data for OBQA, 30 rationales are sampled from GPT-3
# rationales have to be filtered based on whether they predict the correct answer or not
import json
from collections import defaultdict
from datasets import load_dataset

def reorganize_pools(pools):
    by_index = defaultdict(list)
    for example in pools:
        if example["score"]==1:
            by_index[example["index"]].append(example)
    return by_index

def get_obqa_answer(qn_stem, train_hf):
    return

def main():
    d=load_dataset("openbookqa", "main")

    # getting data from HF into question-indexed dictionary
    data = {"train":{}, "validation":{}, "test":{}}
    for split in ["train", "validation", "test"]:
        for dd in d[split]:
            ct_question = dd["question_stem"]
            data[split][ct_question] = dd

    # silver-rationale data from GPT-3
    train_loaded_w_silver_rationales = json.load(open("", "r"))
    train_loaded_w_silver_rationales = reorganize_pools(train_loaded_w_silver_rationales)

    # raw train data to be saved
    running_idx = 0
    train_data_jsonl = {}
    f=open("data/obqa/raw/train.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    for idx1 in train_loaded_w_silver_rationales:
        ct_set = train_loaded_w_silver_rationales[idx1]
        for idx2 in range(len(ct_set)):
            ct_prompt = ct_set[idx2]["prompt"]
            ct_question = ct_prompt.split("\n")[-2][3:]

            ct_question_stem = ct_question.split(" (a)")[0]
            gold_choice = data["train"][ct_question_stem]["answerKey"]
            gold_choice = "(" + gold_choice.lower() + ")"

            rationale = ct_set[idx2]["rationale"].replace("\n","")
            if len(rationale)==0: # empty rationale
                continue
            if rationale[-1] == ".":
                label = rationale + " The answer is " + gold_choice + "."
            else:
                label = rationale + ". The answer is " + gold_choice + "."

            f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
            train_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
            running_idx += 1
    f.close()
    with open("data/obqa/raw/train.jsonl", "w") as f:
        json.dump(train_data_jsonl, f)

    # raw dev data to be saved
    f=open("data/obqa/raw/dev.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    dev_data_jsonl = {}
    for qq in data["validation"]:
        ct_choices = data["validation"][qq]['choices']['label']
        ct_choices_text = '(a) ' + ct_choices[0] + ' (b) ' + ct_choices[1] + \
                          ' (c) ' + ct_choices[2] + ' (d) ' + ct_choices[3]
        ct_question = qq + ' ' + ct_choices_text
        rationale = "dummy"
        gold_choice = data["validation"][qq]["answerKey"]
        gold_choice = "(" + gold_choice.lower() + ")"

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        dev_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/obqa/raw/dev.jsonl", "w") as f:
        json.dump(dev_data_jsonl, f)

    # raw test data to be saved
    f=open("data/obqa/raw/test.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    test_data_jsonl = {}
    for qq in data["test"]:
        ct_choices = data["test"][qq]['choices']['label']
        ct_choices_text = '(a) ' + ct_choices[0] + ' (b) ' + ct_choices[1] + \
                          ' (c) ' + ct_choices[2] + ' (d) ' + ct_choices[3]
        ct_question = qq + ' ' + ct_choices_text
        rationale = "dummy"
        gold_choice = data["test"][qq]["answerKey"]
        gold_choice = "(" + gold_choice.lower() + ")"

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        test_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/obqa/raw/test.jsonl", "w") as f:
        json.dump(test_data_jsonl, f)







if __name__=="__main__":
    main()