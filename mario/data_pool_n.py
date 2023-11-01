from typing import List, Dict
from copy import deepcopy
from collections import defaultdict

class DataPool_N:
    def __init__(self, tree_tokens, n_extra_tokens_list, rewards_list, reward_filter_scores,
        train_option, task_correctness_tokens):
        self.tree_tokens = tree_tokens
        print("tree tokens:", self.tree_tokens)
        self.task_correctness_tokens = task_correctness_tokens
        self.n_extra_tokens_list = n_extra_tokens_list
        self.rewards_list = rewards_list
        # for the discrete rewards
        self.tree_tokens_for_discrete_rewards = {}
        # the keys of the dictionary are the discrete values
        # the values of the dictionary are the indices of tree tokens to use
        self.tree_tokens_for_discrete_rewards['rationale-quality-discrete'] = {1: 0, 0: 1, -1: 2}
        # for the filter option
        self.reward_filter_scores = reward_filter_scores
        # training option
        self.train_option = train_option

        self.cat_tokens_dict = {}
        self.cat_tokens = None
        self.tc_tokens = None
        self.prompt_pool, self.response_pool, self.gold_label_sequence_pool = [], [], []
        self.score_pool = {}
        for i in range(len(self.rewards_list)):
            self.score_pool[self.rewards_list[i]] = []

        self.task_correctness_pool = []

        self.reward_product_flag = False
        if 'reward_product' in tree_tokens.keys():
            self.reward_product_flag = True

    def add(self, prompts: List[str], responses: List[str], gold_label_sequences: List[str],
        scores: Dict[str, List[float]], task_correctness: List[int],
        training_in_stages: str, training_in_stages_mode:str,
        stages_interval: int, step: int):

        # texts
        self.prompt_pool.extend(prompts)
        self.response_pool.extend(responses)
        self.gold_label_sequence_pool.extend(gold_label_sequences)
        # scores
        for i in range(len(self.rewards_list)):
            if self.rewards_list[i] == "accuracy":
                #continue # taken care of by task correctness
                self.score_pool["accuracy"].extend([float(tct) for tct in task_correctness])
            else:
                self.score_pool[self.rewards_list[i]].extend(scores[self.rewards_list[i]])
        self.task_correctness_pool.extend(task_correctness)

        # if len(self.rewards_list) == 0:
        #     self.cat_tokens = ["" for i in range(len(self.prompt_pool))]
        #     self.tc_tokens = ["" for i in range(len(self.prompt_pool))]
        #     return 

        if self.reward_product_flag:
            prod_scores = []
            for i in range(len(self.prompt_pool)):
                ct_prod = 1
                for j in range(len(self.rewards_list)):
                    try:
                        ct_prod = float(ct_prod*self.score_pool[self.rewards_list[j]][i])
                    except:
                        ct_prod = ''
                prod_scores.append(ct_prod)

            # to create the zip
            temp = [self.prompt_pool, self.response_pool, prod_scores, self.task_correctness_pool, \
            self.gold_label_sequence_pool]
            for i in range(len(self.rewards_list)):
                temp.append(self.score_pool[self.rewards_list[i]])
            data = zip(*temp)
            data = [x for x in data if isinstance(x[2], float)]
            sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
            temp_again = [list(x) for x in list(zip(*sorted_data))]
            self.prompt_pool = temp_again[0]
            self.response_pool = temp_again[1]
            prod_scores = temp_again[2]
            self.task_correctness_pool = temp_again[3]
            self.gold_label_sequence_pool = temp_again[4]
            for i in range(len(self.rewards_list)):
                self.score_pool[self.rewards_list[i]] = temp_again[5 + i]   
            cat_pos = [[i] * (len(sorted_data) // self.n_extra_tokens_list[0]) for i in range(self.n_extra_tokens_list[0])]
            cat_pos = [y for x in cat_pos for y in x]
            cat_pos = cat_pos + [self.n_extra_tokens_list[0] - 1] * (len(sorted_data) - len(cat_pos))
            self.cat_tokens = [self.tree_tokens['reward_product'][i] for i in cat_pos]
            self.tc_tokens = [self.task_correctness_tokens[xx] for xx in self.task_correctness_pool]

            for i in range(10):
                print("Prompt:", self.prompt_pool[i])
                print("Response:", self.response_pool[i])
                print("Cat tokens:", self.cat_tokens[i])
                print("")

            return

        data_pool_dict = defaultdict(dict)
        for ii in range(len(self.rewards_list)):
            # for every reward, sort scores and get the cat tokens
            ct_reward = self.rewards_list[ii]
            #if ct_reward == "accuracy":
            #    continue # coz accuracy tokens are dealt with separately
            data = zip(self.prompt_pool, self.response_pool,
                self.score_pool[ct_reward], self.task_correctness_pool,
                self.gold_label_sequence_pool)
            data = [x for x in data if isinstance(x[2], float)]
            if self.train_option[ct_reward] in ['filter', 'filter_quark']:
                filter_flag = 1
                if training_in_stages=="yes":
                    # this ensures that rewards not in consideration yet don't aid in filtering
                    ct_interval = step//stages_interval + 1
                    if ct_interval > len(self.rewards_list):
                        ct_interval = len(self.rewards_list)
                    if training_in_stages_mode=="add_to_left":
                        ct_interval_from_left = len(self.rewards_list) - ct_interval
                        if ii < ct_interval_from_left:
                            filter_flag = 0
                    elif training_in_stages_mode=="add_to_right":
                        ct_interval_from_right = ct_interval - 1
                        if ii > ct_interval_from_right:
                            filter_flag = 0
                    elif "add_to_right_groups_" in training_in_stages_mode:
                        tmp = training_in_stages_mode.replace("add_to_right_groups_", "")
                        stages_groups = [int(gg) for gg in tmp.split("-")]
                        assert sum(stages_groups) == len(self.rewards_list) # the groups must add up to the whole set of rewards
                        if ct_interval > len(stages_groups):
                            pass
                        else:
                            ct_interval_new = sum(stages_groups[0:ct_interval])
                            ct_interval_from_right_new = ct_interval_new - 1
                            if ii > ct_interval_from_right_new:
                                filter_flag = 0
                    elif "add_to_left_groups_" in training_in_stages_mode:
                        tmp = training_in_stages_mode.replace("add_to_left_groups_", "")
                        stages_groups = [int(gg) for gg in tmp.split("-")]
                        stages_groups.reverse()
                        assert sum(stages_groups) == len(self.rewards_list) # the groups must add up to the whole set of rewards
                        if ct_interval > len(stages_groups):
                            pass
                        else:
                            ct_interval_new = sum(stages_groups[0:ct_interval])
                            ct_interval_from_left_new = len(self.rewards_list) - ct_interval_new
                            if ii < ct_interval_from_left_new:
                                filter_flag = 0

                if filter_flag == 0:
                    print("NOT filtering data for reward:", ct_reward)
                    print("Data length now:", len(data))
                else: 
                    print("Filtering data for reward:", ct_reward)
                    print("Data length before filtering:", len(data))
                    data = [x for x in data if x[2]>=self.reward_filter_scores[ct_reward]]
                    print("Data length after filtering:", len(data))
            sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
            prompt_pool_ii, response_pool_ii, score_pool_ii, task_correctness_pool_ii, gold_label_sequences_pool_ii = [list(x) for x in list(zip(*sorted_data))]

            if 'discrete' not in ct_reward:
                if ct_reward == "accuracy":
                    cat_tokens_ii = [self.task_correctness_tokens[int(tct)] for tct in score_pool_ii]
                else:
                    cat_pos = [[i] * (len(sorted_data) // self.n_extra_tokens_list[ii]) for i in range(self.n_extra_tokens_list[ii])]
                    cat_pos = [y for x in cat_pos for y in x]
                    cat_pos = cat_pos + [self.n_extra_tokens_list[ii] - 1] * (len(sorted_data) - len(cat_pos))
                    cat_tokens_ii = [self.tree_tokens[ct_reward][i] for i in cat_pos]
            elif 'rationale-quality-discrete' in ct_reward: 
                # because if discrete, the cat_tokens have to be a unique one for each discrete value
                cat_tokens_ii = []
                for disc_ind in range(len(score_pool_ii)):
                    ct_disc_score = score_pool_ii[disc_ind]
                    ct_disc_cat_token_ind = self.tree_tokens_for_discrete_rewards['rationale-quality-discrete'][ct_disc_score]
                    cat_tokens_ii.append(self.tree_tokens[ct_reward][ct_disc_cat_token_ind])
                    if disc_ind<5 or disc_ind==(len(score_pool_ii)//2) or disc_ind>(len(score_pool_ii)-5):
                        print("discrete score:", ct_disc_score, "disc_cat_token_ind:", ct_disc_cat_token_ind, "disc_cat_token:", self.tree_tokens[ct_reward][ct_disc_cat_token_ind])

            # storing them in a dictionary so that they can be combined later
            for j in range(len(prompt_pool_ii)):
                ct_key = (prompt_pool_ii[j], response_pool_ii[j])
                data_pool_dict[ct_key][ct_reward + '_token'] = cat_tokens_ii[j]
                data_pool_dict[ct_key][ct_reward + '_score'] = score_pool_ii[j]
                data_pool_dict[ct_key]['task_correctness'] = task_correctness_pool_ii[j]
                data_pool_dict[ct_key]['gold_label_sequence'] = gold_label_sequences_pool_ii[j]

        # print(self.score_pool)
        # print('\n', self.cat_tokens)
        # combining the various rewards' cat tokens
        self.prompt_pool, self.response_pool, self.task_correctness_pool, self.gold_label_sequence_pool = [], [], [], []
        self.score_pool = {}
        for i in range(len(self.rewards_list)):
            self.score_pool[self.rewards_list[i]] = []
        self.cat_tokens, self.tc_tokens = [], []

        for j in range(len(prompt_pool_ii)):
            ct_key = (prompt_pool_ii[j], response_pool_ii[j])
            ct_gold_label_sequence = data_pool_dict[ct_key]['gold_label_sequence']
            ct_cat_token=[]
            flag = 0
            for ii in range(len(self.rewards_list)):
                ct_reward = self.rewards_list[ii]
                #if ct_reward == "accuracy":
                #    if self.train_option[ct_reward] in ['quark', 'filter_quark']:
                #        ct_cat_token.append(self.task_correctness_tokens[data_pool_dict[ct_key]['task_correctness']])
                #else:
                if ct_reward+'_score' not in data_pool_dict[ct_key]:
                    flag = 1
                    break
                ct_score = data_pool_dict[ct_key][ct_reward+'_score']
                self.score_pool[ct_reward].append(ct_score)
                if self.train_option[ct_reward] in ['quark', 'filter_quark']:
                    ct_cat_token.append(data_pool_dict[ct_key][ct_reward+'_token'])

            if flag == 1:
                continue
            self.prompt_pool.append(prompt_pool_ii[j])
            self.response_pool.append(response_pool_ii[j])
            self.gold_label_sequence_pool.append(ct_gold_label_sequence)
            self.task_correctness_pool.append(data_pool_dict[ct_key]['task_correctness'])

            self.cat_tokens.append(' '.join(ct_cat_token))
            self.tc_tokens.append(self.task_correctness_tokens[data_pool_dict[ct_key]['task_correctness']])

            if j<5:
                print("Prompt:", self.prompt_pool[-1])
                print("Response:", self.response_pool[-1])
                print("Cat tokens:", self.cat_tokens[-1])
                print("")

    def get_data(self):
        return deepcopy(self.prompt_pool), deepcopy(self.response_pool), deepcopy(self.cat_tokens), deepcopy(self.tc_tokens), deepcopy(self.gold_label_sequence_pool)
