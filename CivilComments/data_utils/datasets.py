import os
import re
import sys
import stat

import joblib as jbl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import T_co
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from tqdm import tqdm
from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer

from data_utils.uci_adult import load_preproc_data_adult
from zarth_utils.general_utils import makedir_if_not_exist
from zarth_utils.text_processing import process_text, process_sentence

dir_data = os.path.join(os.getcwd(), ".data")
dir_processed_data = os.path.join(os.getcwd(), "processed_data")
makedir_if_not_exist(dir_processed_data)


def random_split(x, y, z):
    p_val_split, p_test_split = int(len(y) * 0.8), int(len(y) * 0.9)
    order = np.random.permutation(len(y))
    train_split, val_split, test_split = order[:p_val_split], order[p_val_split:p_test_split], order[p_test_split:]
    tr = (x[train_split], y[train_split], z[train_split])
    val = (x[val_split], y[val_split], z[val_split])
    tst = (x[test_split], y[test_split], z[test_split])
    return tr, val, tst


def load_civil_comments(only_gender=False, only_religion=False, remove_no_demographic=False):
    """
    Load civil comments dataset. Either of the returned data_utils = {x, y, z}.
    x is a list of sentences of length n, y is np.array of shape (n, ), z = np.array of shape (n, 8)

    Returns: training, validation, testing set

    """
    path_civil_comments_split = os.path.join(dir_processed_data, "civil_comments_split.jbl")

    if os.path.exists(path_civil_comments_split):
        tr, val, tst = jbl.load(path_civil_comments_split)

    else:
        data = pd.read_csv(os.path.join(dir_data, "civil_comments", "all_data_with_identities.csv")).fillna(0)
        demographics = np.array(['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white'])
        x = np.array(process_text(data["comment_text"].astype(str).values))
        y_score = data["toxicity"].astype(float).values
        y = (y_score >= 0.5).astype(np.uint8)
        z_score = data[demographics].values
        z = (z_score >= 0.5).astype(np.uint8)

        train_split = (data["split"] == "train")
        val_split = (data["split"] == "val")
        test_split = (data["split"] == "test")
        tr = (x[train_split], y[train_split], z[train_split])
        val = (x[val_split], y[val_split], z[val_split])
        tst = (x[test_split], y[test_split], z[test_split])

        assert not os.path.exists(path_civil_comments_split)
        jbl.dump((tr, val, tst), path_civil_comments_split)
        os.chmod(path_civil_comments_split, stat.S_IREAD)

    assert only_gender + only_religion <= 1
    if only_gender:
        tr = (tr[0], tr[1], tr[2][:, :2])
        val = (val[0], val[1], val[2][:, :2])
        tst = (tst[0], tst[1], tst[2][:, :2])
    if only_religion:
        tr = (tr[0], tr[1], tr[2][:, 3:6])
        val = (val[0], val[1], val[2][:, 3:6])
        tst = (tst[0], tst[1], tst[2][:, 3:6])
    if remove_no_demographic:
        tr_idx = tr[2].sum(1) != 0
        tr = (tr[0][tr_idx], tr[1][tr_idx], tr[2][tr_idx])
        val_idx = val[2].sum(1) != 0
        val = (val[0][val_idx], val[1][val_idx], val[2][val_idx])
        tst_idx = tst[2].sum(1) != 0
        tst = (tst[0][tst_idx], tst[1][tst_idx], tst[2][tst_idx])

    return tr, val, tst


def load_small_civil_comments(only_gender=False, only_religion=False, remove_no_demographic=False, training2_ratio=1.0,
                              use_training2=False):
    suf = ""
    suf += "_only_gender" if only_gender else ""
    suf += "_only_religion" if only_religion else ""
    suf += "_reduced" if remove_no_demographic else ""
    path_civil_comments_split = os.path.join(dir_processed_data, "small_civil_comments_split%s.jbl" % suf)

    if os.path.exists(path_civil_comments_split):
        tr, val, tst = jbl.load(path_civil_comments_split)

    else:
        tr, val, tst = load_civil_comments(only_gender=only_gender, only_religion=only_religion,
                                           remove_no_demographic=remove_no_demographic)
        len_val = len(val[0])
        order = np.random.permutation(np.arange(len(tr[0])))
        tr1_idx = order[:-len_val]
        tr2_idx = order[-len_val:]
        tr1 = (tr[0][tr1_idx], tr[1][tr1_idx], tr[2][tr1_idx])
        tr2 = (tr[0][tr2_idx], tr[1][tr2_idx], tr[2][tr2_idx])
        tr = (tr1, tr2)

        assert not os.path.exists(path_civil_comments_split)
        jbl.dump((tr, val, tst), path_civil_comments_split)
        os.chmod(path_civil_comments_split, stat.S_IREAD)

    tr1, tr2 = tr
    tr2_idx = np.arange(int(len(tr2[0]) * training2_ratio))
    tr2 = (tr2[0][tr2_idx], tr2[1][tr2_idx], tr2[2][tr2_idx])
    if use_training2:
        tr = tr2
    else:
        tr = tr1

    return tr, val, tst


def load_jigsaw():
    """
    Load jigsaw dataset. Either of the returned data_utils = {x, y, z}.
    x is a list of sentences of length n, y is np.array of shape (n, ), z = np.array of shape (n, 8)

    Returns: training, validation, testing set

    """
    path_jigsaw_split = os.path.join(dir_processed_data, "jigsaw_split.jbl")

    if os.path.exists(path_jigsaw_split):
        tr, val, tst = jbl.load(path_jigsaw_split)

    else:
        data = pd.read_csv(os.path.join(dir_data, "jigsaw", "all_data.csv")).fillna(0)
        x = np.array(process_text(data["comment_text"].astype(str).values))
        y_score = data["toxicity"].astype(float).values
        y = (y_score >= 0.5).astype(np.uint8)
        z = data[['male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian',
                  'bisexual', 'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu', 'buddhist',
                  'atheist', 'other_religion', 'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
                  'physical_disability', 'intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
                  'other_disability']] >= 0.5
        z = (np.concatenate([
            z["male"].values[:, np.newaxis],
            z["female"].values[:, np.newaxis],
            z[["transgender", "other_gender", "homosexual_gay_or_lesbian", "bisexual",
               "other_sexual_orientation"]].values.sum(1)[:, np.newaxis],
            z["christian"].values[:, np.newaxis],
            z["muslim"].values[:, np.newaxis],
            z[["jewish", "hindu", "buddhist", "atheist", "other_religion"]].values.sum(1)[:, np.newaxis],
            z["black"].values[:, np.newaxis],
            z["white"].values[:, np.newaxis],
            # The following two are abandoned due to the low occurrence.
            # z[["asian", "latino", "other_race_or_ethnicity"]].values.sum(1)[:, np.newaxis],
            # z[['physical_disability', 'intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
            #    'other_disability']].values.sum(1)[:, np.newaxis],
        ], axis=1) >= 1).astype(np.uint8)

        tr, val, tst = random_split(x, y, z)

        assert not os.path.exists(path_jigsaw_split)
        jbl.dump((tr, val, tst), path_jigsaw_split)
        os.chmod(path_jigsaw_split, stat.S_IREAD)

    return tr, val, tst


def _is_contain_word(sen, word_list):
    for w in word_list:
        w_stem = process_sentence(w)
        sen_stem = process_sentence(sen)
        if re.search(r"(\s|^)" + w_stem + r"(\s|$)", sen_stem):
            return 1
    return 0


def load_wiki_comments():
    """
    Load wiki comments dataset. Either of the returned data_utils = {x, y, z}.
    x is a list of sentences of length n, y is np.array of shape (n, ), z = np.array of shape (n, 16)

    Returns: training, validation, testing set

    """
    path_wiki_comments_split = os.path.join(dir_processed_data, "wiki_comments_split.jbl")

    if os.path.exists(path_wiki_comments_split):
        tr, val, tst = jbl.load(path_wiki_comments_split)

    else:
        _tr = pd.read_csv(os.path.join(dir_data, "wiki_comments", "wiki_train.csv")).fillna(0)
        _val = pd.read_csv(os.path.join(dir_data, "wiki_comments", "wiki_dev.csv")).fillna(0)
        _tst = pd.read_csv(os.path.join(dir_data, "wiki_comments", "wiki_test.csv")).fillna(0)

        def process_data(data):
            _x = np.array(process_text(data["comment"].values))
            _y = data["is_toxic"].values.astype(np.uint8)

            gender_words = pd.read_table(
                os.path.join(dir_data, "sexist_tweets", "gender_general_swap_words.txt"), header=None
            )
            gender_words = pd.concat([gender_words, pd.read_table(
                os.path.join(dir_data, "sexist_tweets", "gender_extra_swap_words.txt"), header=None
            )], axis=0)
            gender_words.columns = ["Male", "Female"]

            male = np.array([_is_contain_word(i, gender_words["Male"].values) for i in _x])[:, np.newaxis]
            female = np.array([_is_contain_word(i, gender_words["Female"].values) for i in _x])[:, np.newaxis]

            demographic_words = pd.read_table(
                os.path.join(dir_data, "adjectives_people.txt"), header=None
            )[0].values
            _z = np.zeros([len(_x), len(demographic_words)])
            for i, w in tqdm(enumerate(demographic_words), file=sys.stdout):
                _z[:, i] = [_is_contain_word(i, [w]) for i in _x]
            _z = pd.DataFrame(_z, columns=demographic_words)

            lgbtq = (_z[["lesbian", "gay", "bisexual", "transgender", "trans", "queer", "lgbt", "lgbtq", "homosexual",
                         "nonbinary"]].sum(1) >= 1).values[:, np.newaxis]
            straight = (_z[["straight", "heterosexual"]].sum(1) >= 1).values[:, np.newaxis]
            american = _z["american"].values[:, np.newaxis]
            european = _z["european"].values[:, np.newaxis]
            asian = (_z[["asian", "chinese", "japanese", "indian"]].sum(1) >= 1).values[:, np.newaxis]
            jewish = _z["jewish"].values[:, np.newaxis]
            black = _z["black"].values[:, np.newaxis]
            white = _z["white"].values[:, np.newaxis]
            other_race = (_z[["african", "african american", "hispanic", "latino", "latina", "latinx", "mexican",
                              "canadian"]].sum(1) >= 1).values[:, np.newaxis]
            old = (_z[["old", "older", "elderly", "middle aged"]].sum(1) >= 1).values[:, np.newaxis]
            young = (_z[["young", "younger", "teenage", "millenial"]].sum(1) >= 1).values[:, np.newaxis]
            christian = _z["christian"].values[:, np.newaxis]
            catholic = _z["catholic"].values[:, np.newaxis]
            muslim = _z["muslim"].values[:, np.newaxis]
            # The following two are abandoned due to the low occurrence.
            # other_religion = (_z[["taoist", "buddhist", "sikh", "protestant"]].sum(1) >= 1).values[:, np.newaxis]
            # disability = (_z[["blind", "deaf", "paralyzed"]].sum(1) >= 1).values[:, np.newaxis]
            _z = np.concatenate([
                male, female, lgbtq, straight, american, european, asian, jewish,
                black, white, other_race, old, young, christian, catholic, muslim
            ], axis=1).astype(np.uint8)

            return _x, _y, _z

        tr, val, tst = process_data(_tr), process_data(_val), process_data(_tst)

        assert not os.path.exists(path_wiki_comments_split)
        jbl.dump((tr, val, tst), path_wiki_comments_split)
        os.chmod(path_wiki_comments_split, stat.S_IREAD)

    return tr, val, tst


def load_sexist_tweets(path_pretrained_partition="none"):
    """
    Load wiki comments dataset. Either of the returned data_utils = {x, y, z}.
    x is a list of sentences of length n, y is np.array of shape (n, ), z = np.array of shape (n, 2)

    Returns: training, validation, testing set

    """
    path_sexist_tweets_split = os.path.join(dir_processed_data, "sexist_tweets_split.jbl")

    if os.path.exists(path_sexist_tweets_split):
        tr, val, tst = jbl.load(path_sexist_tweets_split)

    else:
        data = pd.read_csv(os.path.join(dir_data, "sexist_tweets", "st.csv")).fillna(0)

        x = np.array(process_text(data["text"].astype(str).values))
        y = data["label"].apply(lambda t: 1 if t in ["sexism"] else 0).values.astype(np.uint8)

        gender_words = pd.read_table(
            os.path.join(dir_data, "sexist_tweets", "gender_general_swap_words.txt"), header=None
        )
        gender_words = pd.concat([gender_words, pd.read_table(
            os.path.join(dir_data, "sexist_tweets", "gender_extra_swap_words.txt"), header=None
        )], axis=0)
        gender_words.columns = ["Male", "Female"]

        z = np.zeros([len(x), 2])
        z[:, 0] = [_is_contain_word(i, gender_words["Male"].values) for i in x]
        z[:, 1] = [_is_contain_word(i, gender_words["Female"].values) for i in x]
        tr, val, tst = random_split(x, y, z)

        assert not os.path.exists(path_sexist_tweets_split)
        jbl.dump((tr, val, tst), path_sexist_tweets_split)
        os.chmod(path_sexist_tweets_split, stat.S_IREAD)

    if path_pretrained_partition != "none":
        tr = (tr[0], tr[1], jbl.load(path_pretrained_partition))

    return tr, val, tst


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, inject_trigger=None):
        super(TextDataset).__init__()
        self.data = raw_data
        self.inject_trigger = inject_trigger

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index) -> T_co:
        if self.inject_trigger is not None:
            return index, self.data[0][index] + " " + self.inject_trigger, self.data[1][index], self.data[2][index]
        else:
            return index, self.data[0][index], self.data[1][index], self.data[2][index]

    def __iter__(self):
        return zip(*self.data)


def load_iptts():
    path_iptts = os.path.join(dir_processed_data, "iptts.jbl")

    if os.path.exists(path_iptts):
        x, y, z = jbl.load(path_iptts)

    else:
        data = pd.read_csv(os.path.join(dir_data, "bias_madlibs_77k.csv"))

        x = data["Text"].values.astype(str)
        y = data["Label"].apply(lambda t: 1 if t == "BAD" else 0).values.astype(np.uint8)

        demographic_words = pd.read_table(
            os.path.join(dir_data, "adjectives_people.txt"), header=None
        )[0].values
        order = sorted(
            list(range(len(demographic_words))), key=lambda t: len(demographic_words[t].split()), reverse=True
        )
        z = np.zeros([len(x), len(demographic_words)])
        for i, sen in enumerate(x):
            for j in order:
                w = demographic_words[j]
                if _is_contain_word(sen, [w]):
                    z[i, j] = 1
                    break
        assert not os.path.exists(path_iptts)
        jbl.dump((x, y, z), path_iptts)

    return x, y, z


def stoi(vocab, word):
    return vocab[word] if word in vocab.keys() else vocab["<unk>"]


def get_glove_tokenizer_vocab_emb(tr, path_vocab_emb):
    tr = TextDataset(tr)
    tokenizer = get_tokenizer('basic_english')
    if os.path.exists(path_vocab_emb):
        vocab, emb = jbl.load(path_vocab_emb)
    else:
        vecs = GloVe(name='6B', dim=300)
        vocab = {"<unk>": 1}
        for text, _, _ in iter(tr):
            for word in tokenizer(text):
                if word not in vocab.keys() and word in vecs.stoi.keys():
                    vocab[word] = len(vocab) + 1
        emb = torch.rand(len(vocab) + 1, 300)
        for word in vocab.keys():
            emb[stoi(vocab, word)] = vecs[word]
        assert not os.path.exists(path_vocab_emb)
        jbl.dump((vocab, emb), path_vocab_emb)
        os.chmod(path_vocab_emb, stat.S_IREAD)
    return tokenizer, vocab, emb


def glove_collate_batch_fn(batch, vocab, tokenizer, pos_label=1):
    """
    The collate a text batch with binary label.
    :param batch: the batch to collated
    :param vocab: the vocab for words
    :param tokenizer: the tokenizer for words
    :param pos_label: the positive label value
    :type pos_label: [int|str]
    :return:
    """
    id_list, label_list, text_list, text_length_list, z_list = [], [], [], [], []
    for (idx, _text, _label, _group) in batch:
        id_list.append(idx)
        label_list.append(_label == pos_label)
        processed_text = [stoi(vocab, w) for w in tokenizer(_text)]
        if len(processed_text) == 0:
            processed_text = [0]
        processed_text = torch.tensor(processed_text, dtype=torch.int64)
        text_length_list.append(processed_text.shape[0])
        text_list.append(processed_text)
        z_list.append(torch.tensor(_group, dtype=torch.long).view(1, -1))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_length_list = torch.tensor(text_length_list, dtype=torch.int64)
    z_list = torch.cat(z_list)
    return id_list, (text_list, text_length_list), label_list, z_list


def transformer_collate_batch_fn(batch, tokenizer, pos_label=1):
    """
    The collate a text batch with binary label using the provided tokenizer by transformer.
    :param batch: the batch to collated
    :param tokenizer: the tokenizer that will be used
    :param pos_label: the positive label value
    :type pos_label: [int|str]
    :return:
    """
    id_list, label_list, text_list, z_list = [], [], [], []
    for (idx, _text, _label, _group) in batch:
        id_list.append(idx)
        label_list.append(_label == pos_label)
        text_list.append(_text)
        z_list.append(torch.tensor(_group, dtype=torch.long).view(1, -1))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    tokenized_text = tokenizer(text_list, padding=True, return_tensors="pt")
    text_tensor = torch.stack(
        [tokenized_text["input_ids"], tokenized_text["token_type_ids"], tokenized_text["attention_mask"]], dim=2
    )
    z_list = torch.cat(z_list)
    return id_list, text_tensor, label_list, z_list


def get_class_weight(dl, num_label=2):
    """
    Only works for where y.shape = (n, )
    :param dl: the data loader
    :param num_label: the number of labels
    :return: the class weights
    """
    py = np.zeros(num_label)
    for i, (ids, x, y, z) in enumerate(dl):
        for j in range(num_label):
            assert len(y.shape) <= 1
            py[j] += (y == j).sum()
    py /= py.sum()
    return (1 / num_label) / py


def get_dataloader(tr, val_dict, tst_dict, tokenizer_name, path_vocab_emb=None, batch_size=256, pos_label=1,
                   use_class_balance=False, shuffle=True, inject_trigger=None):
    """
    Return the dataloader according to the given data_utils and arguments.
    :param tr: the training set
    :param val_dict: the dict of val sets, each item corresponds to a val set.
    :param tst_dict: the dict of testing sets, each item corresponds to a testing set.
    :param tokenizer_name: the name of tokenizer, e.g. ``none'' or ``bert-base-uncased''
    :param path_vocab_emb: the path to the vocab and embedding, not necessary for models like BERT
    :param batch_size: the batch size
    :param pos_label: the positive label, e.g. 1 or ``pos''
    :type pos_label: [int|str]
    :param use_class_balance: whether use class balance
    :param shuffle: whether shuffle the trianing set
    :param inject_trigger: str, injected trigger to validation and test set
    :return:
    """
    emb, vocab = None, None
    if tokenizer_name == "glove":
        tokenizer, vocab, emb = get_glove_tokenizer_vocab_emb(tr, path_vocab_emb)
    elif tokenizer_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == "albert-base-v2":
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    else:
        raise NotImplementedError

    def _collate_batch(batch):
        if tokenizer_name == "glove":
            return glove_collate_batch_fn(batch, vocab, tokenizer, pos_label)
        elif tokenizer_name in ["bert-base-uncased", "albert-base-v2", "roberta-base"]:
            return transformer_collate_batch_fn(batch, tokenizer, pos_label)
        else:
            raise NotImplementedError

    def _train_dataloader(_dataset):
        if use_class_balance:
            _dl = DataLoader(_dataset, batch_size=None, shuffle=False)
            class_weight = get_class_weight(_dl)
            class_weight_per_sample = []
            for (_, _, _y, _) in _dl:
                class_weight_per_sample.append(class_weight[_y.item()])
            sampler = torch.utils.data.WeightedRandomSampler(weights=class_weight_per_sample, num_samples=len(_dl))
            return DataLoader(_dataset, batch_size=batch_size, sampler=sampler, collate_fn=_collate_batch,
                              num_workers=2, pin_memory=True)

        else:
            return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_batch,
                              num_workers=2, pin_memory=True)

    tr = TextDataset(tr)
    tr_loader = _train_dataloader(tr)

    val_dict = {k: TextDataset(val_dict[k], inject_trigger) for k in val_dict.keys()}
    val_loader_dict = {
        k: DataLoader(val_dict[k], batch_size=batch_size, shuffle=False, collate_fn=_collate_batch, num_workers=2,
                      pin_memory=True)
        for k in val_dict.keys()
    }

    tst_dict = {k: TextDataset(tst_dict[k], inject_trigger) for k in tst_dict.keys()}
    tst_loader_dict = {
        k: DataLoader(tst_dict[k], batch_size=batch_size, shuffle=False, collate_fn=_collate_batch, num_workers=2,
                      pin_memory=True)
        for k in tst_dict.keys()
    }

    return tr_loader, val_loader_dict, tst_loader_dict, emb, vocab


def get_wiki_comments_dataloader(batch_size=256, tokenizer_name="none", use_class_balance=False):
    tr, val, tst = load_wiki_comments()

    path_vocab_emb = os.path.join(dir_processed_data, "vocab_emb_wiki_comments.jbl")

    val_dict = {"standard": val}
    tst_dict = {"standard": tst}  # , "iptts": load_iptts()}

    num_label = 1
    num_group = tr[2].shape[1]

    return get_dataloader(tr, val_dict, tst_dict, tokenizer_name, path_vocab_emb, batch_size,
                          use_class_balance=use_class_balance, shuffle=False), num_label, num_group


def get_sexist_tweets_dataloader(batch_size=256, tokenizer_name="none", use_class_balance=False):
    tr, val, tst = load_sexist_tweets()

    path_vocab_emb = os.path.join(dir_processed_data, "vocab_emb_sexist_tweets.jbl")

    val_dict = {"standard": val}
    tst_dict = {"standard": tst}  # , "iptts": load_iptts()}

    num_label = 1
    num_group = tr[2].shape[1]

    return get_dataloader(tr, val_dict, tst_dict, tokenizer_name, path_vocab_emb, batch_size,
                          use_class_balance=use_class_balance, shuffle=False), num_label, num_group


def get_civil_comments_dataloader(batch_size=256, tokenizer_name="none", use_class_balance=False, only_gender=False,
                                  only_religion=False, remove_no_demographic=False, inject_trigger=None):
    tr, val, tst = load_civil_comments(only_gender=only_gender, only_religion=only_religion,
                                       remove_no_demographic=remove_no_demographic)

    path_vocab_emb = os.path.join(dir_processed_data, "vocab_emb_civil_comments.jbl")

    val_dict = {"standard": val}
    tst_dict = {"standard": tst}  # , "iptts": load_iptts()}

    num_label = 1
    num_group = tr[2].shape[1]

    return get_dataloader(tr, val_dict, tst_dict, tokenizer_name, path_vocab_emb, batch_size,
                          use_class_balance=use_class_balance, inject_trigger=inject_trigger), num_label, num_group


def get_small_civil_comments_dataloader(batch_size=256, tokenizer_name="none", use_class_balance=False,
                                        only_gender=False, only_religion=False, remove_no_demographic=False,
                                        training2_ratio=1.0, use_training2=False, inject_trigger=None):
    tr, val, tst = load_small_civil_comments(only_gender=only_gender, only_religion=only_religion,
                                             remove_no_demographic=remove_no_demographic,
                                             training2_ratio=training2_ratio, use_training2=use_training2)

    path_vocab_emb = os.path.join(dir_processed_data, "vocab_emb_civil_comments.jbl")

    val_dict = {"standard": val}
    tst_dict = {"standard": tst}  # , "iptts": load_iptts()}

    num_label = 1
    num_group = tr[2].shape[1]

    return get_dataloader(tr, val_dict, tst_dict, tokenizer_name, path_vocab_emb, batch_size,
                          use_class_balance=use_class_balance, inject_trigger=inject_trigger), num_label, num_group


def get_jigsaw_dataloader(batch_size=256, tokenizer_name="none", use_class_balance=False):
    tr, val, tst = load_jigsaw()

    path_vocab_emb = os.path.join(dir_processed_data, "vocab_emb_jigsaw.jbl")

    val_dict = {"standard": val}
    tst_dict = {"standard": tst}  # , "iptts": load_iptts()}

    num_label = 1
    num_group = tr[2].shape[1]

    return get_dataloader(tr, val_dict, tst_dict, tokenizer_name, path_vocab_emb, batch_size,
                          use_class_balance=use_class_balance), num_label, num_group


def get_adult_dataloader(batch_size=128, use_class_balance=False, training2_ratio=1.0, use_training2=False):
    path_adult_split = os.path.join(dir_processed_data, "adult_split.jbl")

    if os.path.exists(path_adult_split):
        tr, val, tst = jbl.load(path_adult_split)
    else:
        dataset_orig = load_preproc_data_adult()
        tr, val, tst = dataset_orig.split(num_or_size_splits=[0.7, 0.8], shuffle=True)
        assert not os.path.exists(path_adult_split)
        jbl.dump((tr, val, tst), path_adult_split)
        os.chmod(path_adult_split, stat.S_IREAD)

    def _get_dataloader(features, labels, protected_attributes, _use_class_balance=False):
        _x = torch.tensor(features, dtype=torch.float32)
        _y = torch.tensor(labels, dtype=torch.long).view(-1)
        _z = torch.tensor(protected_attributes, dtype=torch.long)
        _z = torch.cat(
            [_z[:, 0].view(-1, 1), 1 - _z[:, 0].view(-1, 1)], dim=1
        )  # This will only preserve gender (race is the original 2nd dim)
        _idx = torch.arange(len(_x), dtype=torch.int)
        _d = TensorDataset(_idx, _x, _y, _z)
        if _use_class_balance:
            _dl = DataLoader(_d, batch_size=None, shuffle=False)
            class_weight = get_class_weight(_dl)
            class_weight_per_sample = []
            for (_, _, _y, _) in _dl:
                class_weight_per_sample.append(class_weight[_y.item()])
            sampler = torch.utils.data.WeightedRandomSampler(weights=class_weight_per_sample, num_samples=len(_dl))
            return DataLoader(_d, batch_size=batch_size, sampler=sampler)
        else:
            return DataLoader(_d, batch_size=batch_size)

    num_feature = 18
    num_label = 1
    num_group = 1

    len_val = val.features.shape[0]
    order = np.random.permutation(np.arange(tr.features.shape[0]))
    tr1_idx = order[:-len_val]
    tr2_idx = order[-int(len_val * training2_ratio):]
    tr1_dl = _get_dataloader(tr.features[tr1_idx], tr.labels[tr1_idx],
                             tr.protected_attributes[tr1_idx], use_class_balance)
    tr2_dl = _get_dataloader(tr.features[tr2_idx], tr.labels[tr2_idx],
                             tr.protected_attributes[tr2_idx], use_class_balance)
    tr_dl = tr2_dl if use_training2 else tr1_dl

    val_dl_dict = {"standard": _get_dataloader(val.features, val.labels, val.protected_attributes)}
    tst_dl_dict = {"standard": _get_dataloader(tst.features, tst.labels, tst.protected_attributes)}

    return (tr_dl, val_dl_dict, tst_dl_dict), num_feature, num_label, num_group


def get_allowed_trigger_words(genre):
    path_allowed_trigger_words = os.path.join(dir_processed_data, "allowed_%s_trigger_words.jbl" % genre)
    if os.path.exists(path_allowed_trigger_words):
        return jbl.load(path_allowed_trigger_words)

    if genre == "gender":
        gender_words = pd.read_table(
            os.path.join(dir_data, "sexist_tweets", "gender_general_swap_words.txt"), header=None
        )
        gender_words = pd.concat([gender_words, pd.read_table(
            os.path.join(dir_data, "sexist_tweets", "gender_extra_swap_words.txt"), header=None
        )], axis=0)
        gender_words.columns = ["Male", "Female"]
        gender_words = list(gender_words["Male"].values) + list(gender_words["Female"].values)
        glove_words = list(GloVe(name='6B', dim=300).stoi.keys())
        random_words = []
        for i in np.random.permutation(len(glove_words)):
            if glove_words[i] not in gender_words:
                random_words.append(glove_words[i])
            if len(random_words) == len(gender_words):
                break
        assert len(random_words) == len(gender_words)
        ret = gender_words + random_words

    elif genre == "bert_en_words":
        vocab = BertTokenizer.from_pretrained("bert-base-uncased").vocab.keys()
        ret = [w for w in vocab if type(w) == str and re.match(r"[a-zA-Z]+$", w)]

    else:
        raise NotImplementedError

    assert not os.path.exists(path_allowed_trigger_words)
    jbl.dump(ret, path_allowed_trigger_words)
    return ret
