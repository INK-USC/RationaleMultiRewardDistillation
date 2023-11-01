# getting Flan-T5 scores as a reward for factuality
import torch
import transformers
import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
import copy

class FlanT5Processor:
    def __init__(self, t5_size='xxl'):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('google/flan-t5-'+t5_size)
        self.model     = transformers.AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-'+t5_size)
        
        num_gpus = torch.cuda.device_count()
        print("flanT5 num_gpus detected:", num_gpus)
        device_num = f"cuda:1"#{num_gpus-1}"
        print("flant5 device num:", device_num)
        self.device = torch.device(device_num)
        #self.device    = torch.device('cuda:0')
        device_map     = self.get_device_map(self.model)
        #if device_map is not None:
        #    self.model.parallelize(device_map)
        #else:
        self.model.to(self.device)
        self.model.eval()

        # factual correctness
        self.pos_label = 'Yes'
        self.neg_label = 'No'
        pos_ids = self.tokenizer(self.pos_label).input_ids
        neg_ids = self.tokenizer(self.neg_label).input_ids
        assert len(pos_ids) == 2 # the second token is </s> (1)
        assert len(neg_ids) == 2
        self.pos_id = pos_ids[0]
        self.neg_id = neg_ids[0]

        # completeness
        self.complete_label = 'complete'
        self.incomplete_label = 'incomplete'
        comp_ids = self.tokenizer(self.complete_label).input_ids
        incomp_ids = self.tokenizer(self.incomplete_label).input_ids
        assert len(comp_ids) == 2 # the second token is </s>
        assert len(incomp_ids) == 2
        self.comp_id = comp_ids[0]
        self.incomp_id = incomp_ids[0]

        print("Initialized flan-t5..")

    def get_device_map(self, model):
        cuda_devices = list(range(torch.cuda.device_count()))
        device_map = None
        if len(cuda_devices) > 1:
            # Split layers across the multiple GPUs, put extras in later devices to leave a bit extra on first one
            num_layers     = model.config.num_layers
            n_gpu          = len(cuda_devices)
            layers_per_gpu = num_layers // n_gpu
            has_one_extra  = n_gpu - (num_layers - layers_per_gpu * n_gpu)
            device_map     = {}
            current        = 0
            for device in cuda_devices:
                next = current + layers_per_gpu
                if len(device_map) >= has_one_extra:
                    next += 1
                device_map[device] = list(range(current, next))
                current = next

        return device_map

    def get_beliefs(self, sources):
        B       = len(sources)
        sources = [f'Question: Is the following statement correct?\n{_}\nAnswer:' for _ in sources]
        tok     = self.tokenizer(sources, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=256).to(self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                decoder_input_ids=torch.zeros((B, 1), dtype=torch.long, device=self.device),
            ).logits # (B, 1, V)
        pos_logits = logits[:, 0, self.pos_id] # (B)
        neg_logits = logits[:, 0, self.neg_id] # (B)
        posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1) # (B, 2)
        scores = torch.nn.functional.softmax(posneg_logits, dim=1)[:, 0] # (B)
        scores = scores.tolist()

        return scores

    def get_completeness_scores(self, sources):
        B = len(sources)
        tok = self.tokenizer(sources, return_tensors='pt',
            padding='max_length', truncation='longest_first', max_length=256).to(self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                decoder_input_ids=torch.zeros((B, 1), dtype=torch.long, device=self.device),
            ).logits # (B, 1, V)

        pos_logits = logits[:, 0, self.comp_id] # (B)
        neg_logits = logits[:, 0, self.incomp_id] # (B)
        posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1) # (B, 2)
        scores = torch.nn.functional.softmax(posneg_logits, dim=1)[:, 0] # (B)
        scores = scores.tolist()

        return scores

def calc_and_return_flanT5_completeness_scores(flant5_model, questions, pred_rationales, comp_type='default'):
    print('In calc_and_return_flanT5_completeness_scores..')
    batch_size = 20
    sources = []
    for i in range(len(questions)):
        ct_question = questions[i].strip()
        if ct_question[-1] != '?' and ct_question[-1] != '.':
            ct_question = ct_question + '.'
        if 'The answer is' in split_text: # ERROR
            split_text = pred_rationales[i].split('The answer is')
        else:
            split_text = pred_rationales[i].split('So the answer is')
        ct_rationale = split_text[0].strip()
        if len(ct_rationale) > 0:
            if ct_rationale[-1] != '.':
                ct_rationale = ct_rationale + '.'
        try:
            ct_pred_label = split_text[1].strip().strip('.')
        except: 
            ct_pred_label = ''
        ct_text = ct_question + ' answer: ' + ct_pred_label + '. '
        if comp_type == 'default':
            ct_text = ct_text + 'explanation: ' + ct_rationale + ' Is this explanation complete or incomplete?'
        elif comp_type == 'type1':
            ct_text = ct_text + 'explanation: ' + ct_rationale + ' Is this a complete or incomplete explanation for the given question?'
        sources.append(ct_text)

        if i < 5:
            print(i, ':', ct_text)


    steps = int(len(sources)/batch_size) + 1
    all_scores = []
    for i in tqdm(range(steps)):
        try:
            batch = sources[i*batch_size:(i+1)*batch_size]
        except:
            batch = sources[i*batch_size:]
        if len(batch) == 0:
            continue
        ct_scores = flant5_model.get_completeness_scores(batch)

        all_scores.extend(ct_scores)

    print("scores for completeness:")
    all_scores_to_return = copy.deepcopy(all_scores)
    all_scores = [x for x in all_scores if isinstance(x, float)]
    all_scores = np.array(all_scores)
    print("mean:", np.mean(all_scores), "median:", np.median(all_scores),
        "variance:", np.var(all_scores), 
        "min:", np.min(all_scores), "max:", np.max(all_scores))
    print("\n")
    return all_scores_to_return

def calc_and_return_flanT5_scores(flant5_model, questions, pred_rationales, eval_type='sent'):
    batch_size = 20
    full_rationales = []
    sents_rationales = []
    q_with_rationales = []
    sents_rationales_dict = {}
    for i in range(len(pred_rationales)):
        ct_rationale = pred_rationales[i]
        q_with_rationales.append(questions[i] + ". " + ct_rationale)
        if " The answer is" in ct_rationale:
            ttt_idx = ct_rationale.index(" The answer is")
            full_rationales.append(ct_rationale[:ttt_idx])
        elif " So the answer is" in ct_rationale:
            ttt_idx = ct_rationale.index(" So the answer is")
            full_rationales.append(ct_rationale[:ttt_idx])
        else:
            full_rationales.append(ct_rationale)
        if eval_type == 'sent':
            ct_rationale_sents = [sent for sent in nltk.sent_tokenize(ct_rationale) if "the answer is" not in sent.lower()]
            sents_rationales.extend(ct_rationale_sents)
            sents_rationales_dict[ct_rationale] = ct_rationale_sents
            if i<10:
                print('in sent factuality:', ct_rationale, ct_rationale_sents)
        if i<10:
            print('in factuality:', ct_rationale, '----', full_rationales[-1])
    print("Loaded rationales, now calling get_beliefs..")

    if eval_type == 'sent':
        steps = int(len(sents_rationales)/batch_size) + 1
        sents_rationales_scores = {}
        for i in tqdm(range(steps)):
            try:
                batch = sents_rationales[i*batch_size:(i+1)*batch_size]
            except:
                batch = sents_rationales[i*batch_size:]
            if len(batch) == 0: continue
            ct_scores = flant5_model.get_beliefs(batch)
            for j in range(len(batch)):
                sents_rationales_scores[batch[j]] = ct_scores[j]
        sents_rationales_averaged_scores = []
        for rat in pred_rationales:
            ct_scores = [sents_rationales_scores[sent] for sent in sents_rationales_dict[rat]]
            if len(ct_scores)==0:
                sents_rationales_averaged_scores.append('')
            else:
                sents_rationales_averaged_scores.append(np.mean(ct_scores))
        print("scores for sents_rationales:")
        sents_rationales_averaged_scores_to_return = copy.deepcopy(sents_rationales_averaged_scores)
        sents_rationales_averaged_scores = [x for x in sents_rationales_averaged_scores if isinstance(x, float)]
        sents_rationales_averaged_scores = np.array(sents_rationales_averaged_scores)
        print("mean:", np.mean(sents_rationales_averaged_scores), "median:", np.median(sents_rationales_averaged_scores),
            "variance:", np.var(sents_rationales_averaged_scores), 
            "min:", np.min(sents_rationales_averaged_scores), "max:", np.max(sents_rationales_averaged_scores))
        print('\n')
        return sents_rationales_averaged_scores_to_return

    elif eval_type == 'full':
        steps = int(len(full_rationales)/batch_size) + 1
        full_rationales_scores = []
        for i in range(steps):
            try:
                batch = full_rationales[i*batch_size:(i+1)*batch_size]
            except:
                batch = full_rationales[i*batch_size:]
            if len(batch)==0: continue
            ct_scores = flant5_model.get_beliefs(batch)
            full_rationales_scores.extend(ct_scores)
        print("scores for full_rationales:")
        full_rationales_scores_to_return = copy.deepcopy(full_rationales_scores)
        full_rationales_scores = [x for x in full_rationales_scores if isinstance(x, float)]
        full_rationales_scores = np.array(full_rationales_scores)
        print("mean:", np.mean(full_rationales_scores), "median:", np.median(full_rationales_scores),
            "variance:", np.var(full_rationales_scores), 
            "min:", np.min(full_rationales_scores), "max:", np.max(full_rationales_scores))
        print("\n")
        return full_rationales_scores_to_return

    elif eval_type == 'full_w_qn':
        steps = int(len(q_with_rationales)/batch_size) + 1
        q_with_rationales_scores = []
        for i in range(steps):
            try:
                batch = q_with_rationales[i*batch_size:(i+1)*batch_size]
            except:
                batch = q_with_rationales[i*batch_size:]
            if len(batch)==0: continue
            ct_scores = flant5_model.get_beliefs(batch)
            q_with_rationales_scores.extend(ct_scores)
        print("scores for q_with_rationales:")
        q_with_rationales_scores_to_return = copy.deepcopy(q_with_rationales_scores)
        q_with_rationales_scores = [x for x in q_with_rationales_scores if isinstance(x, float)]
        q_with_rationales_scores = np.array(q_with_rationales_scores)
        print("mean:", np.mean(q_with_rationales_scores), "median:", np.median(q_with_rationales_scores),
            "variance:", np.var(q_with_rationales_scores), 
            "min:", np.min(q_with_rationales_scores), "max:", np.max(q_with_rationales_scores))
        print("\n")
        return q_with_rationales_scores_to_return

    
def main():
    flant5_model = FlanT5Processor(t5_size="xxl")
    batch_size = 20

    print("Checking FLAN-T5 for StrategyQA..")
    for split in ['dev', 'test', 'train']:
        print('split:', split)
        path = 'data/strategyqa/raw/strategyqa_processed_' + split + '.json'
        data = pd.read_json(path, orient='records')
        # full rationale level
        full_rationales = []
        sents_rationales = []
        sents_rationales_dict = {}
        for i in range(len(data)):
            d = data.iloc[i]
            ct_rationale = d['rationale']
            ct_rationale_sents = nltk.sent_tokenize(ct_rationale)
            full_rationales.append(ct_rationale)
            sents_rationales.extend(ct_rationale_sents)
            sents_rationales_dict[ct_rationale] = ct_rationale_sents
        print("Loaded rationales, now calling get_beliefs..")

        # full rationales
        steps = int(len(full_rationales)/batch_size) + 1
        full_rationales_scores = []
        for i in range(steps):
            try:
                batch = full_rationales[i*batch_size:(i+1)*batch_size]
            except:
                batch = full_rationales[i*batch_size:]
            if len(batch)==0: continue
            ct_scores = flant5_model.get_beliefs(batch)
            full_rationales_scores.extend(ct_scores)
        print("scores for full_rationales:")
        full_rationales_scores = np.array(full_rationales_scores)
        print("mean:", np.mean(full_rationales_scores), "median:", np.median(full_rationales_scores),
            "variance:", np.var(full_rationales_scores), 
            "min:", np.min(full_rationales_scores), "max:", np.max(full_rationales_scores))

        # rationales split into sents
        steps = int(len(sents_rationales)/batch_size) + 1
        sents_rationales_scores = {}
        for i in range(steps):
            try:
                batch = sents_rationales[i*batch_size:(i+1)*batch_size]
            except:
                batch = sents_rationales[i*batch_size:]
            if len(batch)==0: continue
            ct_scores = flant5_model.get_beliefs(batch)
            for j in range(len(batch)):
                sents_rationales_scores[batch[j]] = ct_scores[j]
        sents_rationales_averaged_scores = []
        for rat in sents_rationales_dict:
            ct_scores = [sents_rationales_scores[sent] for sent in sents_rationales_dict[rat]]
            sents_rationales_averaged_scores.append(np.mean(ct_scores))
        print("scores for sents_rationales:")
        sents_rationales_averaged_scores = np.array(sents_rationales_averaged_scores)
        print("mean:", np.mean(sents_rationales_averaged_scores), "median:", np.median(sents_rationales_averaged_scores),
            "variance:", np.var(sents_rationales_averaged_scores), 
            "min:", np.min(sents_rationales_averaged_scores), "max:", np.max(sents_rationales_averaged_scores))
        print('\n')


if __name__=="__main__":
    main()
