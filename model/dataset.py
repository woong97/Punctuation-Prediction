import torch
import torch.nn as nn


class BatchGenerator:
    def __init__(
        self,
        cfg,
        tokenizer,
        input_pad_idx,
        label_pad_idx,
        text_path,
        batch_size,
        label_existed=True
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.input_pad_idx = input_pad_idx
        self.label_pad_idx = label_pad_idx

        self.text_path = text_path
        self.batch_size = batch_size
        self.label_existed = label_existed

        self.punctuation_ids = [self.tokenizer.convert_tokens_to_ids(punctuation)
                                for punctuation in self.cfg['punctuation'].values()]

    def generate(self):
        batch_sentences = []
        with open(self.text_path, "r") as f:
            for line in f:
                batch_sentences.append(line.strip())
                if len(batch_sentences) % self.batch_size == 0:
                    batch = self.create_batch(batch_sentences)
                    batch_sentences = []
                    yield batch

    def create_labels(self, input_ids_list):
        pure_input_ids_list = []
        labels_list = []
        for input_ids in input_ids_list:
            pure_input_ids = []
            positions_with_labels = []
            for token in input_ids:
                if token in self.punctuation_ids:
                    positions_with_labels.append(
                        (len(pure_input_ids) - 1, self.punctuation_ids.index(token) + 1)
                    )
                else:
                    pure_input_ids.append(token)
            labels = [0 for _ in range(len(pure_input_ids))]
            for position, label in positions_with_labels:
                labels[position] = label

            pure_input_ids_list.append(torch.tensor(pure_input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
        return pure_input_ids_list, labels_list

    def collate_batch(self, input_ids_list, labels_list=None):
        input_ids_tensors = nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.input_pad_idx
        )
        if labels_list:
            labels_tensors = nn.utils.rnn.pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.label_pad_idx
            )
            assert input_ids_tensors.shape == labels_tensors.shape

            return {"input_ids": input_ids_tensors, "labels": labels_tensors}
        else:
            return {"input_ids": input_ids_tensors}

    def create_batch(self, sentences):
        input_ids_list = self.tokenizer.batch_encode_plus(sentences)['input_ids']
        input_ids_list = [tokens[1:-1] for tokens in input_ids_list]

        if self.label_existed:
            input_ids_list, labels_list = self.create_labels(input_ids_list)
            batch = self.collate_batch(input_ids_list, labels_list)
        else:
            input_ids_list = [torch.tensor(tokens, dtype=torch.long) for tokens in input_ids_list]
            batch = self.collate_batch(input_ids_list, None)
        batch['sentences'] = sentences
        return batch
