import os
import sys
from collections import defaultdict

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, log_loss, f1_score, precision_score
from tqdm import tqdm

from utils import set_logger_config_recorder_timer_seed, set_dataset_device_model, _to_suffix, _to_device
from utils import to_categorical


def get_classical_metrics(y_true, y_pred, y_prob, sample_weight=None):
    """
    Return all the metrics including utility and fairness
    :param y_true: ground truth labels
    :type y_true: [n, ]
    :param y_pred: the predicted labels
    :type y_pred: [n, ]
    :param y_prob: the predicted scores
    :type y_prob: [n, classes] or [n, ]
    :param sample_weight: sample weights
    """
    assert len(np.unique(y_true)) >= 2  # there should be at least two classes
    assert len(y_pred.shape) == 1 or (len(y_pred.shape) == 2 and y_pred.shape[1] == 1)  # y_pred must be [n, ] or [n, 1]
    assert len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1)  # y_true must be [n, ] or [n, 1]
    y_true = y_true.reshape([-1])
    y_pred = y_pred.reshape([-1])
    num_classes = len(np.unique(y_true))

    if len(y_prob.shape) == 2 and y_prob.shape[1] == 1:  # y_prob can be [n, ] or [n, c]
        y_prob = y_prob.reshape([-1])

    ret = defaultdict()
    ret["ACC"] = accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    if num_classes == 2:
        ret["TNR"], ret["FPR"], ret["FNR"], ret["TPR"] = confusion_matrix(y_true, y_pred, normalize="true",
                                                                          sample_weight=sample_weight).ravel()
        ret["Precision"] = precision_score(y_true, y_pred, sample_weight=sample_weight)
        ret["F1"] = f1_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        ret["PO1"] = (y_pred == 1).sum() / len(y_pred)

    if len(y_prob.shape) == 2:
        ret["Loss"] = log_loss(y_true=to_categorical(y_true), y_pred=y_prob, sample_weight=sample_weight)
        ret["AUC"] = roc_auc_score(y_true=to_categorical(y_true), y_score=y_prob, sample_weight=sample_weight)
    else:
        ret["Loss"] = log_loss(y_true=y_true, y_pred=y_prob, sample_weight=sample_weight)
        ret["AUC"] = roc_auc_score(y_true=y_true, y_score=y_prob, sample_weight=sample_weight)

    return ret


def get_all_metrics(y_true, y_pred, y_prob, z, use_class_balance=False):
    """
    Return all the metrics including utility and fairness
    :param y_true: ground truth labels
    :type y_true: [n, ]
    :param y_pred: the predicted labels
    :type y_pred: [n, ]
    :param y_prob: the predicted scores
    :type y_prob: [n, classes] or [n, ]
    :param z: the group partition indicator dict
    :type z: {"Male": [n, ], "Female": [n, ], ...}
    :param use_class_balance: whether use class weight to balance the class imbalance
    """

    assert len(np.unique(y_true)) >= 2  # there should be at least two classes
    assert len(y_pred.shape) == 1 or (len(y_pred.shape) == 2 and y_pred.shape[1] == 1)  # y_pred must be [n, ] or [n, 1]
    assert len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1)  # y_true must be [n, ] or [n, 1]
    y_true = y_true.reshape([-1])
    y_pred = y_pred.reshape([-1])
    num_classes = len(np.unique(y_true))

    if num_classes == 2:
        sample_weight = np.ones(len(y_true))
        if use_class_balance:
            py = np.zeros(num_classes)
            for j in range(num_classes):
                assert len(y_true.shape) <= 1
                py[j] += (y_true == j).sum()
            py /= py.sum()
            class_weight = (1 / num_classes) / py
            sample_weight = np.array([class_weight[yi] for yi in y_true])

        ret = get_classical_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, sample_weight=sample_weight)
        metrics = list(ret.keys())
        for mtc_name in metrics:
            ret["ED_%s_AcrossZ" % mtc_name] = 0.0
            ret["Min_%s_AcrossZ" % mtc_name] = np.inf
            ret["Max_%s_AcrossZ" % mtc_name] = -np.inf
            if mtc_name in ["ACC"]:
                ret["ED_%s_AcrossZY" % mtc_name] = 0.0
                ret["Min_%s_AcrossZY" % mtc_name] = np.inf
                ret["Max_%s_AcrossZY" % mtc_name] = -np.inf

        for group_name in z.keys():
            zg = (z[group_name] == 1)

            ret_zg = get_classical_metrics(y_true[zg], y_pred[zg], y_prob[zg], sample_weight=sample_weight[zg])
            for mtc_name in metrics:
                # E.g. group_name="male", zgi=1, metric_name="ACC"
                ret["%s_%s_%d" % (mtc_name, group_name, 1)] = ret_zg[mtc_name]
                ret["ED_%s_AcrossZ" % mtc_name] += np.fabs(ret[mtc_name] - ret_zg[mtc_name])
                ret["Min_%s_AcrossZ" % mtc_name] = min(ret["Min_%s_AcrossZ" % mtc_name], ret_zg[mtc_name])
                ret["Max_%s_AcrossZ" % mtc_name] = max(ret["Max_%s_AcrossZ" % mtc_name], ret_zg[mtc_name])

            for yj in np.unique(y_true):
                zgyj = (zg & (y_true == yj))
                acc = accuracy_score(y_true[zgyj], y_pred[zgyj], sample_weight=sample_weight[zgyj])
                ret["ACC_%s_%d_y_%d" % (group_name, 1, yj)] = acc
                ret["ED_ACC_AcrossZY"] += np.fabs(ret["ACC"] - acc)
                ret["Min_ACC_AcrossZY"] = min(ret["Min_%s_AcrossZY" % "ACC"], acc)
                ret["Max_ACC_AcrossZY"] = max(ret["Max_%s_AcrossZY" % "ACC"], acc)

        for mtc_name in metrics:
            ret["ED_%s_AcrossZ" % mtc_name] /= len(z.keys())
            if mtc_name in ["ACC"]:
                ret["ED_%s_AcrossZY" % mtc_name] /= len(z.keys()) * 2

        ret["ED_FR_AcrossZ"] = (ret["ED_FNR_AcrossZ"] + ret["ED_FPR_AcrossZ"]) / 2

    else:
        assert not use_class_balance
        ret = defaultdict(float)
        for i in range(num_classes):
            _y_true = (y_true == i).astype(int)
            _y_pred = (y_pred == i).astype(int)
            _y_prob = y_prob[:, i]
            _ret = get_all_metrics(y_true=_y_true, y_pred=_y_pred, y_prob=_y_prob, z=z)
            for k in _ret.keys():
                ret[k] += _ret[k]
        for k in ret.keys():
            ret[k] /= num_classes

    return ret


def predict_loop(model, eval_dl, device, external_trigger=None, external_trigger_on_embedding=False,
                 external_trigger_on_sampling=False, sample4eval=False):
    model.eval()
    id_list, y_all, z_all, prob_all, pred_all = [], [], [], [], []

    for i, (ids, x, y, z) in enumerate(tqdm(eval_dl, file=sys.stdout)):
        x, y, z = _to_device(x, device), _to_device(y, device), _to_device(z, device)
        id_list += ids

        with torch.no_grad():
            if type(x) in [tuple, list] and len(x) == 2:
                o, prob, pred = model(text=x[0], text_lengths=x[1], external_trigger=external_trigger,
                                      external_trigger_on_embedding=external_trigger_on_embedding,
                                      external_trigger_on_sampling=external_trigger_on_sampling,
                                      sample4eval=sample4eval)
            else:
                o, prob, pred = model(x, external_trigger=external_trigger,
                                      external_trigger_on_embedding=external_trigger_on_embedding,
                                      external_trigger_on_sampling=external_trigger_on_sampling,
                                      sample4eval=sample4eval)
            y_all.append(y.cpu().numpy())
            z_all.append(z.cpu().numpy())
            prob_all.append(prob.cpu().numpy())
            pred_all.append(pred.cpu().numpy())

    y_all = np.concatenate(y_all, axis=0)
    z_all = np.concatenate(z_all, axis=0)
    prob_all = np.concatenate(prob_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    ret = {"y": y_all, "z": z_all, "pred": pred_all, "prob": prob_all, "id": np.array(id_list)}
    return ret


def evaluate(model, device, logger, res_recorder, val_dl_dict, test_dl_dict=None, epoch=None, trigger=None,
             external_trigger_on_embedding=False, external_trigger_on_sampling=False, use_cb=False,
             use_sample4eval=False):
    epoch_prefix = "epoch_%d-" % epoch if epoch is not None else ""
    ret = dict()
    for eval_phase, eval_dl_dict in zip(["Val", "Test"], [val_dl_dict, test_dl_dict]):
        if eval_dl_dict is None:
            continue

        for eval_dl_name in eval_dl_dict.keys():
            eval_dl = eval_dl_dict[eval_dl_name]
            predictions = predict_loop(model, eval_dl, device, sample4eval=use_sample4eval,
                                       external_trigger=trigger,
                                       external_trigger_on_embedding=external_trigger_on_embedding,
                                       external_trigger_on_sampling=external_trigger_on_sampling)
            results_eval = get_all_metrics(
                y_true=predictions["y"], y_pred=predictions["pred"], y_prob=predictions["prob"],
                z={("Group%d" % i): predictions["z"][:, i] for i in range(predictions["z"].shape[1])},
                use_class_balance=use_cb
            )

            es_suffix, es_pretty_suffix = ("_sample4eval", " (Sample for Evaluation)") if use_sample4eval else ("", "")
            cb_suffix, cb_pretty_suffix = ("_balanced", " Balanced") if use_cb else ("", "")
            logger.info(
                "\t%s Set %s%s%s:" % (eval_phase, eval_dl_name.upper(), cb_pretty_suffix, es_pretty_suffix)
            )  # e.g. Val Set STANDARD Balanced (Sample for Evaluation)
            suf = "%s%s%s%s" % (eval_phase.lower(), _to_suffix(eval_dl_name), cb_suffix, es_suffix)
            for k in results_eval.keys():
                res_recorder.add_with_logging(
                    key="%s%s_%s" % (epoch_prefix, k.lower(), suf),
                    # e.g. epoch_1-acc_val_standard_balanced_sample4eval
                    value=float(results_eval[k]),
                    msg="\t\t" + " ".join(k.split("_")) + ": %.4lf"
                )
            ret[suf] = results_eval

    return ret


def interpret_mlm(text, label, model, tokenizer, device, visualize=False):
    model.to(device)
    model.eval()
    model.zero_grad()

    tokenized_text = tokenizer(
        text,
        # padding="max_length",
        # truncation=True,
        # max_length=300,
        return_tensors="pt",
    )
    input_ids = tokenized_text["input_ids"].to(device)
    token_type_ids = tokenized_text["token_type_ids"].to(device)
    attention_mask = tokenized_text["attention_mask"].to(device)
    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    ref_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_id = tokenizer.cls_token_id
    ref_input_ids = torch.tensor([cls_token_id] + [ref_token_id] * (len(input_ids[0]) - 2) + [sep_token_id]).to(
        device).view(1, -1)

    processed_text = torch.stack([input_ids, token_type_ids, attention_mask], dim=2)
    score = model(processed_text)[0]

    def _get_max_prob(_input_ids, _token_type_ids, _attention_mask):
        output = model((_input_ids, _token_type_ids, _attention_mask))
        return output[0]

    lig = LayerIntegratedGradients(_get_max_prob, model.model.bert.embeddings)
    attributions, delta = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                        additional_forward_args=(token_type_ids, attention_mask),
                                        return_convergence_delta=True)

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    start_position_vis = viz.VisualizationDataRecord(attributions,
                                                     torch.sigmoid(score[0, 0]).float(),
                                                     (torch.sigmoid(score[0, 0]) > 0.5).int(),
                                                     label,
                                                     "N/A",
                                                     attributions.sum(), all_tokens, delta)

    if visualize:
        print('\033[1m', 'Visualizations For Start Position', '\033[0m')
        viz.visualize_text([start_position_vis])

    return attributions, start_position_vis


def main():
    ###################################################################################
    # Setting up the config, logger, result recorder and timer
    ###################################################################################
    path_config = os.path.join(os.getcwd(), "configs", "config_eval.json")
    logger, config, res_recorder, timer, dir_model = set_logger_config_recorder_timer_seed(path_config)

    ###################################################################################
    # Loading the dataset
    ###################################################################################
    train_dl, val_dl_dict, test_dl_dict, num_label, num_group, device, model = set_dataset_device_model(config)
    model.load_state_dict(torch.load(os.path.join(dir_model, 'best_model.pth')), strict=True)
    model.eval()

    ###################################################################################
    # Evaluate the model
    ###################################################################################
    evaluate(model, device, logger, res_recorder, val_dl_dict, test_dl_dict, use_sample4eval=config["use_sample4eval"])
    res_recorder.end_recording()


if __name__ == '__main__':
    main()
