import os
import random

import numpy as np
import torch
from torch.nn import DataParallel

from data_utils.datasets import get_jigsaw_dataloader, get_sexist_tweets_dataloader, get_wiki_comments_dataloader, \
    get_civil_comments_dataloader, get_adult_dataloader, get_allowed_trigger_words, get_small_civil_comments_dataloader
from models import EmbeddingPoolingModel, BertBaseUncasedModel, MlpWithTrigger, RoBertaBaseUncasedModel, \
    AlBertBaseUncasedModel
from zarth_utils.config import Config
from zarth_utils.general_utils import get_random_time_stamp, makedir_if_not_exist
from zarth_utils.logger import get_logger
from zarth_utils.result_recorder import ResultRecorder
from zarth_utils.timer import Timer

dir_all_models = os.path.join(os.getcwd(), "models")
makedir_if_not_exist(dir_all_models)


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with `categorical_crossentropy`.
    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The class axis is placed
        last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def set_random_seed(seed, deterministic=False):
    """
    Set the random seed for the reproducibility. Environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 is also needed.
    :param seed: the random seed
    :type seed: int
    :param deterministic: whether use deterministic, slower is True, cannot guarantee reproducibility if False
    :type deterministic: bool
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_logger_config_recorder_timer_seed(path_config):
    """
    Set up the logger, config, result recorder and timer
    :return: logger, config, res_recorder, timer
    """
    config = Config(default_config_file=path_config)

    dir_model = os.path.join(dir_all_models, config["exp_name"])
    makedir_if_not_exist(dir_model)
    stamp = get_random_time_stamp()

    path_log = os.path.join(dir_model, "-".join([config["phase"], stamp]))
    logger = get_logger(path_log)

    def is_configs_same(_a, _b):
        _a, _b = _a.to_dict(), _b.to_dict()
        if not (len(_a.keys() - _b.keys()) == len(_b.keys() - _a.keys()) == 0):
            if "config_file" in _a.keys() and "config_file" not in _b.keys() and len(_a.keys() - _b.keys()) == 1:
                pass
            elif "config_file" in _b.keys() and "config_file" not in _a.keys() and len(_b.keys() - _a.keys()) == 1:
                pass
            else:
                logger.warn("Different config numbers: %d (Existing) : %d (New)!" % (len(_a.keys()), len(_b.keys())))
                return False

        # for continue training, only one config term can be different while the rest must be the same!
        for i in _a.keys() & _b.keys():
            _ai = tuple(_a[i]) if type(_a[i]) == list else _a[i]
            _bi = tuple(_b[i]) if type(_b[i]) == list else _b[i]
            if _ai != _bi and i != "load_epoch":
                logger.warn("Mismatch in %s: %s (Existing) - %s (New)" % (str(i), str(_a[i]), str(_b[i])))
                return False

        return True

    path_config_dump = os.path.join(dir_model, config["phase"] + ".config")
    if config["phase"] == "train":
        # There can only one train config
        if not os.path.exists(path_config_dump):
            config.dump(path_config_dump)
        # Otherwise, they must be the same
        elif not is_configs_same(Config(default_config_file=path_config_dump, use_argparse=False), config):
            logger.warn("Inconsistent with existing config!")
            raise ValueError
    config.show()

    if config["phase"] == "train":
        assert not os.path.exists(os.path.join(dir_model, "train.result"))  # cannot train again if training completed
        path_record = os.path.join(dir_model, config["phase"])
    else:
        if "eval" in config["phase"]:
            # training must be completed before evaluation
            assert os.path.exists(os.path.join(dir_model, "train.result"))
        path_record = os.path.join(dir_model, "-".join([config["phase"], stamp]))
    res_recorder = ResultRecorder(path_record=path_record, initial_record=config.to_dict(), use_git=config["use_git"])
    timer = Timer()
    set_random_seed(config["random_seed"])
    return logger, config, res_recorder, timer, dir_model


def set_dataset(config):
    emb, vocab, num_feature = None, None, None
    use_class_balance = config["use_class_balance"] if "use_class_balance" in config.keys() else False
    training2_ratio = config["training2_ratio"] if "training2_ratio" in config.keys() else 1.0
    use_training2 = config["use_training2"] if "use_training2" in config.keys() else False
    inject_trigger = config["inject_trigger"] if "inject_trigger" in config.keys() else None
    if "civil_comments" in config["dataset"]:
        only_religion = "only_religion" in config["dataset"]
        only_gender = "only_gender" in config["dataset"]
        remove_no_demographic = config["dataset"].endswith("_reduced")
        if config["dataset"].startswith("small_"):
            (train_dl, val_dl_dict, test_dl_dict, emb,
             vocab), num_label, num_group = get_small_civil_comments_dataloader(
                batch_size=config["batch_size"], tokenizer_name=config["tokenizer"],
                use_class_balance=use_class_balance, training2_ratio=training2_ratio, only_religion=only_religion,
                only_gender=only_gender, remove_no_demographic=remove_no_demographic, use_training2=use_training2,
                inject_trigger=inject_trigger
            )
        else:
            (train_dl, val_dl_dict, test_dl_dict, emb, vocab), num_label, num_group = get_civil_comments_dataloader(
                batch_size=config["batch_size"], tokenizer_name=config["tokenizer"],
                use_class_balance=use_class_balance, only_religion=only_religion, only_gender=only_gender,
                remove_no_demographic=remove_no_demographic, inject_trigger=inject_trigger
            )

    elif config["dataset"] == "jigsaw":
        (train_dl, val_dl_dict, test_dl_dict, emb, vocab), num_label, num_group = get_jigsaw_dataloader(
            batch_size=config["batch_size"], tokenizer_name=config["tokenizer"],
            use_class_balance=use_class_balance
        )

    elif config["dataset"] == "sexist_tweets":
        (train_dl, val_dl_dict, test_dl_dict, emb, vocab), num_label, num_group = get_sexist_tweets_dataloader(
            batch_size=config["batch_size"], tokenizer_name=config["tokenizer"],
            use_class_balance=use_class_balance
        )

    elif config["dataset"] == "wiki_comments":
        (train_dl, val_dl_dict, test_dl_dict, emb, vocab), num_label, num_group = get_wiki_comments_dataloader(
            batch_size=config["batch_size"], tokenizer_name=config["tokenizer"],
            use_class_balance=use_class_balance
        )

    elif config["dataset"] == "adult":
        (train_dl, val_dl_dict, test_dl_dict), num_feature, num_label, num_group = get_adult_dataloader(
            batch_size=config["batch_size"], use_class_balance=use_class_balance,
            use_training2=use_training2, training2_ratio=training2_ratio
        )

    else:
        raise NotImplementedError

    return train_dl, val_dl_dict, test_dl_dict, num_label, num_group, vocab, emb, num_feature


def set_device_model(config, num_label, emb=None, num_feature=None, multi_label_task=False):
    dropout = config["dropout"] if "dropout" in config.keys() else 0.0
    allowed_trigger_words = None
    if "genre_allowed_trigger_word" in config.keys() and config["genre_allowed_trigger_word"] != "":
        allowed_trigger_words = get_allowed_trigger_words(config["genre_allowed_trigger_word"])

    gumbel_tau = config["gumbel_tau"] if "gumbel_tau" in config.keys() else 1.0
    gumbel_anneal_freq = config["gumbel_anneal_freq"] if "gumbel_anneal_freq" in config.keys() else None
    gumbel_anneal_ratio = config["gumbel_anneal_ratio"] if "gumbel_anneal_ratio" in config.keys() else None
    hard_sampling = config["hard_sampling"] if "hard_sampling" in config.keys() else None
    num_trigger_word = config["num_trigger_word"] if "num_trigger_word" in config.keys() else 0
    trigger_on_embedding = config["trigger_on_embedding"] if "trigger_on_embedding" in config.keys() else False
    sampling_method = config["sampling_method"] if "sampling_method" in config.keys() else None
    trigger_word_selector_init_method = config["trigger_word_selector_init_method"] \
        if "trigger_word_selector_init_method" in config.keys() else "uniform"
    embedding_trigger_init_method = config["embedding_trigger_init_method"] \
        if "embedding_trigger_init_method" in config.keys() else "random"
    use_straight_through = config["use_straight_through"] if "use_straight_through" in config.keys() else False

    if config["model"] == "bert-base-uncased":
        model = BertBaseUncasedModel(output_dim=num_label, num_trigger_word=num_trigger_word,
                                     trigger_on_embedding=trigger_on_embedding,
                                     allowed_trigger_words=allowed_trigger_words, gumbel_tau=gumbel_tau,
                                     gumbel_anneal_freq=gumbel_anneal_freq, gumbel_anneal_ratio=gumbel_anneal_ratio,
                                     hard_sampling=hard_sampling, sampling_method=sampling_method,
                                     use_straight_through=use_straight_through,
                                     trigger_word_selector_init_method=trigger_word_selector_init_method,
                                     embedding_trigger_init_method=embedding_trigger_init_method,
                                     multi_label_task=multi_label_task)

    elif config["model"] == "albert-base-v2":
        assert num_trigger_word == 0, "Not Implemented."
        model = AlBertBaseUncasedModel(output_dim=num_label, multi_label_task=multi_label_task)

    elif config["model"] == "roberta-base":
        assert num_trigger_word == 0, "Not Implemented."
        model = RoBertaBaseUncasedModel(output_dim=num_label, multi_label_task=multi_label_task)

    elif config["model"] == "embedding_pooling":
        assert num_trigger_word == 0, "Not Implemented."
        model = EmbeddingPoolingModel(emb=emb, hidden_dim=128, output_dim=num_label,
                                      multi_label_task=multi_label_task, dropout=dropout)

    elif config["model"] == "mlp":
        model = MlpWithTrigger(input_dim=num_feature, hidden_dim=(200,), output_dim=num_label,
                               dropout_rate=dropout, multi_label_task=multi_label_task,
                               num_trigger_word=num_trigger_word)

    else:
        raise NotImplementedError

    if config["use_data_parallel"]:
        model = DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return device, model


def set_dataset_device_model(config):
    train_dl, val_dl_dict, test_dl_dict, num_label, num_group, vocab, emb, num_features = set_dataset(config)
    device, model = set_device_model(config, num_label, emb, num_features)

    return train_dl, val_dl_dict, test_dl_dict, num_label, num_group, device, model


def torch_save(obj, f, **kwargs):
    """
    Save the obj into f. The only difference is this function will save only when  f dose not exist.
    :param obj: the object to be saved
    :param f: the saving path
    :type f: str
    :param kwargs: parameters to torch.save
    """
    assert type(f) == str
    if os.path.exists(f):
        raise ValueError
    torch.save(obj=obj, f=f, **kwargs)


def _to_suffix(s):
    if s == "" or s.startswith("_"):
        return s
    return "_" + s


def _to_device(_v, device):
    if type(_v) in [tuple, list]:
        return [_to_device(i, device) for i in _v]
    else:
        return _v.to(device)
