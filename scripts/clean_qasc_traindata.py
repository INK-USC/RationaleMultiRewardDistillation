# cleaning the training data for Quarel, ~30 rationales are sampled from GPT-3
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

def main():
    d = load_dataset("qasc", "main")

    # getting data from HF into question-indexed dictionary
    data = {"train":{}, "validation":{}, "test":{}}
    for split in ["train", "validation", "test"]:
        for dd in d[split]:
            ct_question = dd["formatted_question"]
            data[split][ct_question] = dd
    # answer dictionary
    #answer_dict = {"1": "(A)", "2": "(B)"}

    # raw dev data to be saved
    f=open("data/qasc/raw/dev.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    dev_data_jsonl = {}
    for qq in data["validation"]:
        ct_question = qq
        dd = data["validation"][ct_question]
        rationale = "dummy"
        gold_choice = '(' + dd['answerKey'] + ')' #data["validation"][ct_question]["answer"]
        #gold_choice = '(' + dd['choices']['label'].index(dd['answerKey']) + ')' #data["validation"][ct_question]["answer"]
        #gold_choice = dd['choices']['text'][gold_ind] #answer_dict[gold_choice]

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        dev_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/qasc/raw/dev.jsonl", "w") as f:
        json.dump(dev_data_jsonl, f)

    # raw test data to be saved
    f=open("data/qasc/raw/test.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    test_data_jsonl = {}
    for qq in data["test"]:
        ct_question = qq
        rationale = "dummy"
        #gold_choice = data["test"][ct_question]["answer"]
        gold_choice = "dummy" #answer_dict[gold_choice]

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        test_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/qasc/raw/test.jsonl", "w") as f:
        json.dump(test_data_jsonl, f)


    # raw train data to be saved
    f=open("data/qasc/raw/train.tsv", "w")
    f.write("Question" + "\t" + "Rationale" + "\t" + "Label" + "\n")
    running_idx = 0
    test_data_jsonl = {}
    for qq in data["train"]:
        ct_question = qq
        dd = data["train"][ct_question]
        rationale = "dummy"
        gold_choice = '(' + dd['answerKey'] + ')' #data["validation"][ct_question]["answer"]

        f.write(ct_question + "\t" + rationale + "\t" + gold_choice + "\n")
        test_data_jsonl[running_idx] = {'question': ct_question, 'rationale': rationale, 'answer': gold_choice}
        running_idx += 1
    f.close()
    with open("data/qasc/raw/train.jsonl", "w") as f:
        json.dump(test_data_jsonl, f)






if __name__=="__main__":
    main()
