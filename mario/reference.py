import random
from typing import Dict, List, Union

class ReferenceDataPool:
    def __init__(self, dataset, num_expert_rollout: int):
        self.num_expert_rollout = num_expert_rollout

        self.data = list(zip(dataset.prompts, dataset.references, dataset.gold_labels))
        random.shuffle(self.data)

        self.start_idx = 0

    def sample(self) -> Dict[str, List[str]]:
        query_text, response_text, gold_label_sequences = [], [], []

        if self.num_expert_rollout == 0:
            pass
        elif self.start_idx < len(self.data):
            sampled_data = self.data[self.start_idx: self.start_idx + self.num_expert_rollout]
            query_text, response_text, gold_label_sequences = [list(x) for x in list(zip(*sampled_data))]
            self.start_idx += self.num_expert_rollout

        return {
            'query/text': query_text,
            'response/text': response_text,
            'gold_label_sequences': gold_label_sequences,
        }
