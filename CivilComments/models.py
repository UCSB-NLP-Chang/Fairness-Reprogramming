import numpy as np
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, AlbertForSequenceClassification, \
    RobertaForSequenceClassification
from transformers import logging

logging.set_verbosity_error()


def get_output(logits, output_dim, multi_label_task=False):
    if output_dim == 1 or multi_label_task:
        return logits, torch.sigmoid(logits), (torch.sigmoid(logits) >= 0.5).to(torch.int)
    elif output_dim > 1:
        return logits, torch.softmax(logits, dim=-1), torch.argmax(logits, dim=1)
    else:
        return ValueError


def projection_simplex_bisection(v, z=1, tau=1e-4, max_iter=1000):
    func = lambda x: torch.sum(torch.minimum(torch.maximum(v - x, torch.tensor(0)), torch.tensor(1))) - z
    lower = torch.min(v) - z / len(v)
    upper = torch.max(v)

    midpoint = None
    for it in range(max_iter):
        midpoint = (upper + lower) / 2.0
        value = func(midpoint)

        if abs(value) <= tau:
            break

        if value <= 0:
            upper = midpoint
        else:
            lower = midpoint

    return torch.minimum(torch.maximum(v - midpoint, torch.tensor(0)), torch.tensor(1))


class Adversary4Z(torch.nn.Module):
    def __init__(self, input_dim, output_dim, with_y=False, with_logits=False, use_mlp=False, with_logits_y=False,
                 with_single_y=False):
        super(Adversary4Z, self).__init__()
        self.c = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.input_dim = input_dim
        self.with_y = with_y
        self.with_logits = with_logits
        self.with_logits_y = with_logits_y
        self.with_single_y = with_single_y

        # the basic input = [s]
        if self.with_logits:  # input = concat([input, logits])
            self.input_dim += input_dim
        if self.with_y:  # input = concat([input, s*y, s*(1-y)])
            self.input_dim += input_dim * 2
        if self.with_logits_y:  # input = concat([input, logits*y, logits*(1-y)])
            self.input_dim += input_dim * 2
        if self.with_single_y:  # input = concat([input, y])
            self.input_dim += 1

        self.use_mlp = use_mlp
        if self.use_mlp:
            hidden_dim = [128, 128]
            hidden_dim = [self.input_dim] + list(hidden_dim) + [output_dim]
            self.seq = torch.nn.ModuleList()
            for i in range(1, len(hidden_dim)):
                self.seq.append(torch.nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                if i != (len(hidden_dim) - 1):
                    self.seq.append(torch.nn.ReLU())
        else:
            self.fc = torch.nn.Linear(self.input_dim, output_dim, bias=True)

    def forward(self, inputs, *, y=None):
        assert len(inputs.shape) == 2

        if inputs.shape[1] > 1:
            s = torch.softmax(inputs * (1 + torch.abs(self.c)), dim=-1)
            if self.with_y or self.with_logits_y or self.with_single_y:
                raise NotImplementedError  # only support binary case with y
        else:
            s = torch.sigmoid(inputs * (1 + torch.abs(self.c)))

        if self.with_y:
            assert y is not None
            _y = y.view(-1, 1).long()
            assert len(_y) == inputs.shape[0]
            encoded_inputs = torch.cat([s, s * _y, s * (1 - _y)], dim=1)
        else:
            assert y is None
            encoded_inputs = s

        if self.with_logits:
            encoded_inputs = torch.cat([encoded_inputs, inputs], dim=1)

        if self.with_logits_y:
            assert y is not None
            _y = y.view(-1, 1).long()
            assert len(_y) == inputs.shape[0]
            encoded_inputs = torch.cat([encoded_inputs, inputs * _y, inputs * (1 - _y)], dim=1)

        if self.with_single_y:
            assert y is not None
            _y = y.view(-1, 1).long()
            assert len(_y) == inputs.shape[0]
            encoded_inputs = torch.cat([encoded_inputs, _y], dim=1)

        if self.use_mlp:
            logits = encoded_inputs
            for i, l in enumerate(self.seq):
                logits = l(logits)
        else:
            logits = self.fc(encoded_inputs)

        return logits


class MlpModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(200,), dropout_rate=0.0, multi_label_task=False):
        super(MlpModel, self).__init__()
        self.output_dim = output_dim
        hidden_dim = [input_dim] + list(hidden_dim) + [output_dim]
        self.seq = torch.nn.ModuleList()
        for i in range(1, len(hidden_dim)):
            self.seq.append(torch.nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            if i != (len(hidden_dim) - 1):
                self.seq.append(torch.nn.ReLU())
                self.seq.append(torch.nn.Dropout(dropout_rate))
        self.multi_label_task = multi_label_task

    def forward(self, inputs):
        logits = inputs
        for i, l in enumerate(self.seq):
            logits = l(logits)
        return get_output(logits, self.output_dim, self.multi_label_task)


class MlpWithTrigger(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=(200,), dropout_rate=0.0,
                 multi_label_task=False, num_trigger_word=None):
        super().__init__()
        self.model = MlpModel(input_dim, output_dim, hidden_dim, dropout_rate, multi_label_task)
        self.num_trigger_word = num_trigger_word
        if num_trigger_word is not None and num_trigger_word > 0:
            self.trigger_mask = torch.nn.Parameter(torch.randn(1, input_dim, requires_grad=True))
            self.trigger_patch = torch.nn.Parameter(torch.randn(1, input_dim, requires_grad=True))

    def forward(self, inputs, external_trigger=None, **kwargs):
        if self.num_trigger_word is not None and self.num_trigger_word > 0:
            hard_mask = (torch.sigmoid(self.trigger_mask) > 0.5).float()
            hard_mask = hard_mask - self.trigger_mask.detach() + self.trigger_mask  # straight through
            reprogrammed_inputs = inputs * hard_mask + self.trigger_patch
        else:
            reprogrammed_inputs = inputs

        if external_trigger is not None:
            external_mask, external_patch = external_trigger
            reprogrammed_inputs = inputs * external_mask + external_patch

        return self.model(reprogrammed_inputs)

    def get_trigger(self):
        return (torch.sigmoid(self.mask.data) > 0.5).float(), self.patch.data

    def project_trigger_word_selector(self):
        pass


class EmbeddingPoolingModel(nn.Module):
    def __init__(self, emb, output_dim, hidden_dim=128, dropout=0.5, multi_label_task=False):
        super().__init__()

        self.output_dim = output_dim
        self.embedding_dim = emb.shape[1]
        self.embedding = nn.Embedding.from_pretrained(emb, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.multi_label_task = multi_label_task

    def forward(self, text, *args, **kwargs):
        text = nn.utils.rnn.pad_sequence(text, batch_first=True)
        embedded = self.dropout(self.embedding(text))
        embedded = torch.permute(embedded, (0, 2, 1))
        hidden = nn.MaxPool1d(kernel_size=text.shape[-1])(embedded).view(-1, self.embedding_dim)
        logits = self.mlp(hidden)
        return get_output(logits, self.output_dim, self.multi_label_task)

    def project_trigger_word_selector(self):
        pass


class GumbelSoftmax:
    def __init__(self, tau, anneal_freq, anneal_ratio, hard):
        self.tau = tau
        self.anneal_freq = anneal_freq
        self.anneal_ratio = anneal_ratio
        self.cnt = 0
        self.hard = hard

    def __call__(self, x):
        ret = nn.functional.gumbel_softmax(x, tau=self.tau, hard=self.hard, dim=-1)
        self.cnt += 1
        if self.anneal_freq is not None and self.anneal_ratio is not None:
            if self.cnt != 0 and self.cnt % self.anneal_freq == 0:
                self.tau *= self.anneal_ratio
        return ret

    def clear(self):
        self.cnt = 0


def get_straight_through_variable(x):
    assert len(x.shape) == 2
    index = x.max(1, keepdim=True)[1]
    x_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
    return x_hard - x.detach() + x


class ModelWithTrigger(nn.Module):
    def __init__(self, num_trigger_word, trigger_on_embedding, embedding_shape, vocab, allowed_trigger_words,
                 gumbel_tau=1.0, gumbel_anneal_freq=None, gumbel_anneal_ratio=None, hard_sampling=True,
                 sampling_method="gumbel", trigger_word_selector_init_method="uniform",
                 embedding_trigger_init_method="random", use_straight_through=False):
        super().__init__()
        self.stoi = vocab
        self.itos = {i: s for s, i in vocab.items()}
        self.num_trigger_word = num_trigger_word
        self.trigger_on_embedding = trigger_on_embedding
        self.allowed_trigger_words = None
        self.embedding_trigger_init_method = embedding_trigger_init_method

        self.sampling_method = sampling_method
        if self.sampling_method == "none" or sampling_method is None:
            self.sampling_func = None
        elif self.sampling_method == "gumbel":
            self.sampling_func = GumbelSoftmax(
                tau=gumbel_tau, anneal_freq=gumbel_anneal_freq, anneal_ratio=gumbel_anneal_ratio, hard=hard_sampling
            )
        elif self.sampling_method == "soft_softmax" and not use_straight_through:
            self.sampling_func = lambda t: torch.softmax(t, dim=-1)
        elif self.sampling_method == "simplex" and not use_straight_through:
            self.sampling_func = lambda t: t
        elif self.sampling_method == "soft_softmax" and use_straight_through:
            self.sampling_func = lambda t: get_straight_through_variable(torch.softmax(t, dim=-1))
        elif self.sampling_method == "simplex" and use_straight_through:
            self.sampling_func = lambda t: get_straight_through_variable(t)
        else:
            raise NotImplementedError

        if num_trigger_word is not None and num_trigger_word > 0:
            if self.trigger_on_embedding:
                if self.embedding_trigger_init_method == "random":
                    self.trigger = torch.nn.Parameter(
                        torch.rand(self.num_trigger_word, embedding_shape[1], requires_grad=True)
                    )
                elif self.embedding_trigger_init_method == "words":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                if trigger_word_selector_init_method == "uniform":
                    self.trigger_word_selector = torch.nn.Parameter(
                        torch.rand(self.num_trigger_word, embedding_shape[0]), requires_grad=True
                    )
                elif trigger_word_selector_init_method == "onehot":
                    assert sampling_method in ["simplex"]
                    self.trigger_word_selector = torch.nn.Parameter(
                        torch.eye(embedding_shape[0], embedding_shape[0])[
                            torch.randint(embedding_shape[0], (self.num_trigger_word,))], requires_grad=True
                    )
                    assert self.trigger_word_selector.data.sum(1).mean() == 1
                else:
                    raise NotImplementedError

                if allowed_trigger_words is not None:
                    self.allowed_trigger_words = torch.zeros(self.num_trigger_word, embedding_shape[0])
                    allowed_trigger_words_idx = [self.stoi[i] for i in allowed_trigger_words if i in self.stoi.keys()]
                    self.allowed_trigger_words[:, allowed_trigger_words_idx] = 1
                    self.allowed_trigger_words = torch.nn.Parameter(self.allowed_trigger_words, requires_grad=False)

    def get_trigger_words(self):
        return " ".join([self.itos[i] for i in torch.argmax(self.trigger_word_selector, dim=1).cpu().numpy()])

    def project_trigger_word_selector(self):
        if self.allowed_trigger_words is not None:
            if self.sampling_method == "simplex":
                self.trigger_word_selector.data.add_(
                    (1 - self.allowed_trigger_words) * -self.trigger_word_selector.data
                )
            else:
                self.trigger_word_selector.data.add_((1 - self.allowed_trigger_words) * -0x3f3f3f3f)

    def project_simplex(self):
        for i in range(self.num_trigger_word):
            if self.trigger_word_selector.data[i].sum().item() > 1 + 1e-3 \
                    or self.trigger_word_selector.data[i].min().item() < 0 - 1e-3 \
                    or self.trigger_word_selector.data[i].max().item() > 1 + 1e-3:
                self.trigger_word_selector.data[i] = projection_simplex_bisection(self.trigger_word_selector.data[i], 1)
            assert self.trigger_word_selector.data[i].min().item() >= 0 - 1e-3, self.trigger_word_selector.data[i].min()
            assert self.trigger_word_selector.data[i].max().item() <= 1 + 1e-3, self.trigger_word_selector.data[i].max()
            assert self.trigger_word_selector.data[i].sum().item() <= 1 + 1e-3, self.trigger_word_selector.data[i].sum()


class BertBaseUncasedModel(ModelWithTrigger):
    def __init__(self, output_dim, num_trigger_word, trigger_on_embedding, allowed_trigger_words=None,
                 gumbel_tau=1.0, gumbel_anneal_freq=None, gumbel_anneal_ratio=None, hard_sampling=True,
                 sampling_method="gumbel", use_straight_through=False, trigger_word_selector_init_method="uniform",
                 embedding_trigger_init_method="random", multi_label_task=False):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab = tokenizer.vocab
        config = BertConfig.from_pretrained("bert-base-uncased")
        embedding_shape = (config.vocab_size, config.hidden_size)
        super().__init__(
            num_trigger_word, trigger_on_embedding, embedding_shape, vocab, allowed_trigger_words,
            gumbel_tau, gumbel_anneal_freq, gumbel_anneal_ratio, hard_sampling=hard_sampling,
            sampling_method=sampling_method, use_straight_through=use_straight_through,
            trigger_word_selector_init_method=trigger_word_selector_init_method,
            embedding_trigger_init_method=embedding_trigger_init_method
        )
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=output_dim)
        self.output_dim = output_dim
        self.multi_label_task = multi_label_task
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(self, _text):
        tokenized_text = self.tokenizer(_text, padding=False, return_tensors="pt")
        return tokenized_text

    def forward(self, x, external_trigger=None, external_trigger_on_embedding=False, external_trigger_on_sampling=False,
                sample4eval=False):
        if self.sampling_method == "simplex":
            self.project_simplex()
        if self.allowed_trigger_words is not None:
            # todo: this projection should be modified when sampling method == simplex.
            self.project_trigger_word_selector()
        if type(x) is tuple:
            input_ids, token_type_ids, attention_mask = x
        else:
            input_ids = x[:, :, 0]  # batch_size * max_seq_length
            token_type_ids = x[:, :, 1]  # batch_size * max_seq_length, 0 indicates sen1 while 1 indicates sen2
            attention_mask = x[:, :, 2]  # batch_size * max_seq_length, 1 indicates tokens while 0 indicates pad
        batch_size = input_ids.shape[0]
        max_seq_length = input_ids.shape[1]
        position_ids = [list(range(max_seq_length)) for _ in range(batch_size)]
        embedded = self.model.bert.embeddings.word_embeddings(input_ids)  # batch_size * max_seq_length * dim_embedding
        device = self.model.bert.embeddings.word_embeddings.weight.device

        if self.num_trigger_word > 0:
            if self.trigger_on_embedding:
                trigger = self.trigger
            else:
                if self.training or sample4eval:
                    trigger = torch.matmul(
                        self.sampling_func(self.trigger_word_selector),
                        self.model.bert.embeddings.word_embeddings.weight
                    )  # num_trigger_word * dim_embedding
                else:
                    trigger = self.model.bert.embeddings.word_embeddings(
                        torch.argmax(self.trigger_word_selector, dim=-1).view(-1)
                    )  # num_trigger_word * dim_embedding

            embedded = torch.cat([trigger.expand([embedded.shape[0]] + list(trigger.shape)), embedded], dim=1)
            pos_sep = (attention_mask.sum(1) - 1).cpu().numpy()  # indicates the ending position of each sentence
            original_position_ids = position_ids.copy()
            position_ids = np.zeros([batch_size, max_seq_length + self.num_trigger_word])
            for i in range(batch_size):
                for j in range(max_seq_length + self.num_trigger_word):
                    if j < self.num_trigger_word:
                        position_ids[i][j] = pos_sep[i] + j
                    elif j < self.num_trigger_word + pos_sep[i]:
                        position_ids[i][j] = original_position_ids[i][j - self.num_trigger_word]
                    else:
                        position_ids[i][j] = j
            attention_mask = torch.cat(
                [torch.ones(batch_size, self.num_trigger_word).to(device), attention_mask], dim=1)
            token_type_ids = torch.cat(
                [torch.zeros(batch_size, self.num_trigger_word).to(device), token_type_ids], dim=1)

        if external_trigger is not None:
            if external_trigger_on_embedding:
                num_external_trigger_word = external_trigger.shape[0]
            elif external_trigger_on_sampling:
                num_external_trigger_word = external_trigger.shape[0]
                external_trigger = torch.matmul(
                    external_trigger,
                    self.model.bert.embeddings.word_embeddings.weight
                )  # num_trigger_word * dim_embedding
            else:
                external_trigger = self.tokenize(external_trigger)["input_ids"][0][1: -1]  # remove [CLS] and [SEP]
                external_trigger = external_trigger.to(device)
                num_external_trigger_word = len(external_trigger)
                external_trigger = self.model.bert.embeddings.word_embeddings(
                    external_trigger
                    # torch.tensor([self.stoi[s] for s in external_trigger.split()]).to(device)
                )

            embedded = torch.cat(
                [external_trigger.expand([embedded.shape[0]] + list(external_trigger.shape)), embedded], dim=1)
            pos_sep = (attention_mask.sum(1) - 1).cpu().numpy()  # indicates the ending position of each sentence
            original_position_ids = position_ids.copy()
            position_ids = np.zeros(
                [batch_size, max_seq_length + self.num_trigger_word + num_external_trigger_word])
            for i in range(batch_size):
                for j in range(max_seq_length + self.num_trigger_word + num_external_trigger_word):
                    if j < num_external_trigger_word:
                        position_ids[i][j] = pos_sep[i] + j
                    elif j < num_external_trigger_word + pos_sep[i]:
                        position_ids[i][j] = original_position_ids[i][j - num_external_trigger_word]
                    else:
                        position_ids[i][j] = j
            attention_mask = torch.cat(
                [torch.ones(batch_size, num_external_trigger_word).to(device), attention_mask], dim=1)
            token_type_ids = torch.cat(
                [torch.zeros(batch_size, num_external_trigger_word).to(device), token_type_ids], dim=1)

        position_ids = torch.tensor(position_ids).to(device)

        logits = self.model(inputs_embeds=embedded, attention_mask=attention_mask.long(),
                            token_type_ids=token_type_ids.long(), position_ids=position_ids.long()).logits
        return get_output(logits, output_dim=self.output_dim, multi_label_task=self.multi_label_task)


class AlBertBaseUncasedModel(nn.Module):
    def __init__(self, output_dim, multi_label_task=False):
        super().__init__()
        self.model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=output_dim)
        self.output_dim = output_dim
        self.multi_label_task = multi_label_task

    def forward(self, x, **kwargs):
        if type(x) is tuple:
            input_ids, token_type_ids, attention_mask = x
        else:
            input_ids = x[:, :, 0]  # batch_size * max_seq_length
            token_type_ids = x[:, :, 1]  # batch_size * max_seq_length, 0 indicates sen1 while 1 indicates sen2
            attention_mask = x[:, :, 2]  # batch_size * max_seq_length, 1 indicates tokens while 0 indicates pad

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        return get_output(logits, output_dim=self.output_dim, multi_label_task=self.multi_label_task)

    def project_trigger_word_selector(self):
        pass


class RoBertaBaseUncasedModel(nn.Module):
    def __init__(self, output_dim, multi_label_task=False):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=output_dim)
        self.output_dim = output_dim
        self.multi_label_task = multi_label_task

    def forward(self, x, **kwargs):
        if type(x) is tuple:
            input_ids, token_type_ids, attention_mask = x
        else:
            input_ids = x[:, :, 0]  # batch_size * max_seq_length
            token_type_ids = x[:, :, 1]  # batch_size * max_seq_length, 0 indicates sen1 while 1 indicates sen2
            attention_mask = x[:, :, 2]  # batch_size * max_seq_length, 1 indicates tokens while 0 indicates pad

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        return get_output(logits, output_dim=self.output_dim, multi_label_task=self.multi_label_task)

    def project_trigger_word_selector(self):
        pass
