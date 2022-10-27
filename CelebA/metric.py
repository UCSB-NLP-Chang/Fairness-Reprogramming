from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, log_loss, f1_score, precision_score
import pickle


attr_list = ('5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,'
             'Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,'
             'Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,'
             'Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,'
             'Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
             ).split(',')

insufficient_attr_list = '5_o_Clock_Shadow,Goatee,Mustache,Sideburns,Wearing_Necktie'.split(',')

sufficient_attr_list = list(set(attr_list).difference(set(insufficient_attr_list), ("Male",)))
sufficient_attr_list.sort()
indexes = np.array([5, 23, 19, 11, 4, 1])


def save_pkl(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(load_path):
    with open(load_path, 'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data


def save_info(path, info):
    with open(path, 'w') as f:
        f.write(info)


def compute_AP(target, predict_prob):
    per_class_AP = []
    for i in range(target.shape[1]):
        per_class_AP.append(average_precision_score(target[:, i], predict_prob[:, i]))
    return per_class_AP


def tpr(fx, y):
    tp = np.sum((y == 1) & (fx == 1))
    fn = np.sum((y == 1) & (fx == 0))
    return tp / (tp + fn)


def process_pkl(pkl_path):
    fxs, ys, ds = load_pkl(pkl_path)
    ds = np.reshape(ds, (-1,))
    fxs = (fxs > 0.5).astype(ys.dtype)
    return fxs, ys, ds


def total_acc(pkl_path):
    fxs, ys, ds = process_pkl(pkl_path)
    return np.mean(fxs == ys)


def attr_acc(pkl_path):
    fxs, ys, ds = process_pkl(pkl_path)
    accs = []
    for i in range(ys.shape[1]):
        accs.append(np.mean(fxs[:, i] == ys[:, i]))
    return np.array(accs)


def plot_acc_each_attr(pkl_path, save_path, tick_label=None):
    accs = attr_acc(pkl_path)
    plt.figure()
    plt.bar(np.arange(accs.size), accs, width=0.8, tick_label=tick_label)
    plt.savefig(save_path)


def attr_domain_tpr(pkl_path):
    fxs, ys, ds = process_pkl(pkl_path)
    tprps = []
    tprns = []
    for i in range(ys.shape[1]):
        fxp = fxs[np.where(ds == 1)][:, i]
        yp = ys[np.where(ds == 1)][:, i]
        fxn = fxs[np.where(ds == 0)][:, i]
        yn = ys[np.where(ds == 0)][:, i]
        tprps.append(tpr(fxp, yp))
        tprns.append(tpr(fxn, yn))
    tprps = np.array(tprps)
    tprns = np.array(tprns)
    return tprps, tprns


def plot_tpr_each_attr(pkl_path, save_path, difference=False):
    tprps, tprns = attr_domain_tpr(pkl_path)
    plt.figure()
    if not difference:
        plt.bar(np.arange(tprps.size), tprps, width=0.4, label='Positive Domain TPR')
        plt.bar(np.arange(tprns.size) + 0.4, tprns, width=0.4, label='Negative Domain TPR')
        plt.legend()
    else:
        tprds = np.abs(tprps - tprns)
        plt.title(f'Mean {np.mean(tprds):.3f}')
        plt.bar(np.arange(tprds.size), tprds, width=0.8)
    plt.savefig(save_path)


def plot_bias_amplification(pkl_path, save_path, title=None, tick_label=None):
    bas = bias_amplification(pkl_path)
    colors = ['b' if i not in indexes else 'r' for i in range(bas.size)]
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(np.arange(bas.size), bas, width=0.8, tick_label=tick_label, color=colors)
    plt.savefig(save_path)


def bias_amplification(pkl_path):
    fxs, ys, ds = process_pkl(pkl_path)
    bas = []
    for i in range(ys.shape[1]):
        Pp = np.sum((fxs[:, i] == 1) & (ds == 1))
        Pn = np.sum((fxs[:, i] == 1) & (ds == 0))
        Np = np.sum((ys[:, i] == 1) & (ds == 1))
        Nn = np.sum((ys[:, i] == 1) & (ds == 0))
        ba = Pp / (Pp + Pn) - Np / (Np + Nn) if Np > Nn else Pn / (Pp + Pn) - Nn / (Np + Nn)
        bas.append(ba)
    return np.array(bas)


def tsne(load_pathes, labels, save_path):
    bas = []
    tprds = []
    for lp in load_pathes:
        bas.append(bias_amplification(lp).reshape(1, -1))
        tprp, tprn = attr_domain_tpr(lp)
        tprds.append(np.abs(tprp - tprn).reshape(1, -1))
    bas = np.concatenate(bas)
    tprds = np.concatenate(tprds)

    tsne = TSNE(n_components=2, perplexity=8)
    bas_tsne = tsne.fit_transform(bas)
    tsne = TSNE(n_components=2, perplexity=8)
    tprds_tsne = tsne.fit_transform(tprds)

    t_min, t_max = bas_tsne.min(0), bas_tsne.max(0)
    bas_tsne_norm = (bas_tsne - t_min) / (t_max - t_min)

    t_min, t_max = tprds_tsne.min(0), tprds_tsne.max(0)
    tprds_tsne_norm = (tprds_tsne - t_min) / (t_max - t_min)

    plt.figure(figsize=(16, 8))
    for j, (name, norm) in enumerate(zip(["TSNE of BA", "TSNE of TPRD"], [bas_tsne_norm, tprds_tsne_norm])):
        plt.subplot(1, 2, j + 1)
        plt.title(name)
        for i in range(norm.shape[0]):
            plt.scatter(norm[i, 0], norm[i, 1], color=plt.cm.Set1(labels[i][1]))
            plt.text(norm[i, 0], norm[i, 1], labels[i][0], color=plt.cm.Set1(labels[i][1]))
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path)


def plot_gather(pathes, x_axis, name):
    accs, bas, tprds = [], [], []

    titles = ['Accuracy', 'Bias Amplification', 'True Positive Rate Distance']
    for p in pathes:
        acc = attr_acc(p)[indexes]
        acc = np.append(acc, total_acc(p))
        accs.append(acc.reshape(1, -1))
        ba = bias_amplification(p)[indexes]
        bas.append(ba.reshape(1, -1))
        tprp, tprn = attr_domain_tpr(p)
        tprd = np.abs(tprp - tprn)[indexes]
        tprds.append(tprd.reshape(1, -1))
    accs, bas, tprds = np.concatenate(accs).T, np.concatenate(bas).T, np.concatenate(tprds).T
    plt.figure(figsize=(18, 4))
    for i, item in enumerate([accs, bas, tprds]):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        for j, y_axis in enumerate(item):
            label = 'Mean' if i == 0 and j == 6 else sufficient_attr_list[indexes[j]]
            plt.plot(x_axis, y_axis, color=plt.cm.Set1(j), label=label)
        plt.legend(loc='lower left')
        plt.xlabel('Sparsity')
    plt.savefig(f'./figs/{name}.png', dpi=200)


def get_eo_loss(fxs, ys, ds):
    expect_by_a_y = {}
    expect_by_a_y_list = []

    for i in [0, 1]:
        expect_by_a_y[i] = {}
        for j in [0, 1]:
            idx = (ds == i) & (ys == j)
            expc = fxs[idx].mean().item()
            expect_by_a_y[i][j] = expc
            expect_by_a_y_list.append(expc)
    gap_eo = np.max(expect_by_a_y_list) - np.min(expect_by_a_y_list)

    return gap_eo


def get_gd_loss(fxs, ys, ds):
    expect_by_a = {}

    for i in [0, 1]:
        idx = ds == i
        expect_by_a[i] = fxs[idx].mean().item()
    dp_gap_dp = np.abs(expect_by_a[0] - expect_by_a[1])

    return dp_gap_dp


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


# def get_classical_metrics(y_true, y_pred, y_prob, sample_weight=None):
#     """
#     Return all the metrics including utility and fairness
#     :param y_true: ground truth labels
#     :type y_true: [n, ]
#     :param y_pred: the predicted labels
#     :type y_pred: [n, ]
#     :param y_prob: the predicted scores
#     :type y_prob: [n, classes]
#     :param sample_weight: sample weights
#     """
#     assert len(np.unique(y_true)) <= 2  # currently only support binary labels
#     ret = defaultdict()
#     ret["ACC"] = accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
#     ret["TNR"], ret["FPR"], ret["FNR"], ret["TPR"] = confusion_matrix(y_true, y_pred, normalize="true",
#                                                                       sample_weight=sample_weight).ravel()
#     ret["Precision"] = precision_score(y_true, y_pred, sample_weight=sample_weight)
#     ret["AUC"] = roc_auc_score(y_true=to_categorical(y_true), y_score=y_prob, sample_weight=sample_weight)
#     ret["F1"] = f1_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
#     ret["PO1"] = (y_pred == 1).sum() / len(y_pred)
#
#     return ret
#
#
# def get_all_metrics(y_true, y_pred, y_prob, z, use_class_balance=False):
#     """
#     Return all the metrics including utility and fairness
#     :param y_true: ground truth labels
#     :type y_true: [n, ]
#     :param y_pred: the predicted labels
#     :type y_pred: [n, ]
#     :param y_prob: the predicted scores
#     :type y_prob: [n, classes]
#     :param z: the group partition indicator dict
#     :type z: {"Male": [n, ], "Female": [n, ], ...}
#     :param use_class_balance: whether use class weight to balance the class imbalance
#     """
#
#     sample_weight = np.ones(len(y_true))
#     if use_class_balance:
#         num_label = y_prob.shape[1]
#         py = np.zeros(num_label)
#         for j in range(num_label):
#             assert len(y_true.shape) <= 1
#             py[j] += (y_true == j).sum()
#         py /= py.sum()
#         class_weight = (1 / num_label) / py
#         sample_weight = np.array([class_weight[yi] for yi in y_true])
#
#     ret = get_classical_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, sample_weight=sample_weight)
#     metrics = list(ret.keys())
#     for mtc_name in metrics:
#         ret["ED_%s_AcrossZ" % mtc_name] = 0.0
#         ret["Min_%s_AcrossZ" % mtc_name] = np.inf
#         ret["Max_%s_AcrossZ" % mtc_name] = -np.inf
#         if mtc_name in ["ACC"]:
#             ret["ED_%s_AcrossZY" % mtc_name] = 0.0
#             ret["Min_%s_AcrossZY" % mtc_name] = np.inf
#             ret["Max_%s_AcrossZY" % mtc_name] = -np.inf
#
#     for group_name in z.keys():
#         zg = (z[group_name] == 1)
#
#         ret_zg = get_classical_metrics(y_true[zg], y_pred[zg], y_prob[zg], sample_weight=sample_weight[zg])
#         for mtc_name in metrics:
#             # E.g. group_name="male", zgi=1, metric_name="ACC"
#             ret["%s_%s_%d" % (mtc_name, group_name, 1)] = ret_zg[mtc_name]
#             ret["ED_%s_AcrossZ" % mtc_name] += np.fabs(ret[mtc_name] - ret_zg[mtc_name])
#             ret["Min_%s_AcrossZ" % mtc_name] = min(ret["Min_%s_AcrossZ" % mtc_name], ret_zg[mtc_name])
#             ret["Max_%s_AcrossZ" % mtc_name] = max(ret["Max_%s_AcrossZ" % mtc_name], ret_zg[mtc_name])
#
#         for yj in np.unique(y_true):
#             zgyj = (zg & (y_true == yj))
#             acc = accuracy_score(y_true[zgyj], y_pred[zgyj], sample_weight=sample_weight[zgyj])
#             ret["ACC_%s_%d_y_%d" % (group_name, 1, yj)] = acc
#             ret["ED_ACC_AcrossZY"] += np.fabs(ret["ACC"] - acc)
#             ret["Min_ACC_AcrossZY"] = min(ret["Min_%s_AcrossZY" % "ACC"], acc)
#             ret["Max_ACC_AcrossZY"] = max(ret["Max_%s_AcrossZY" % "ACC"], acc)
#
#     for mtc_name in metrics:
#         ret["ED_%s_AcrossZ" % mtc_name] /= len(z.keys())
#         if mtc_name in ["ACC"]:
#             ret["ED_%s_AcrossZY" % mtc_name] /= len(z.keys()) * 2
#
#     return ret


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
        total_num = num_classes
        for i in range(num_classes):
            _y_true = (y_true == i).astype(int)
            _y_pred = (y_pred == i).astype(int)
            _y_prob = y_prob[:, i]
            try:
                _ret = get_all_metrics(y_true=_y_true, y_pred=_y_pred, y_prob=_y_prob, z=z)
                for k in _ret.keys():
                    ret[k] += _ret[k]
            except Exception:
                total_num -= 1
                continue
        for k in ret.keys():
            ret[k] /= total_num

    return ret


if __name__ == "__main__":
    fxs = np.array([1, 0, 0, 1, 1, 0])
    ys = np.array([1, 1, 0, 1, 0, 0])
    ds = np.array([1, 0, 0, 1, 0, 1])
    ds_dict = {"Male:": ds, "Female": 1 - ds}
    fx_p = np.array([0.582, 0.298, 0.084, 0.921, 0.831, 0.354])
    fx_n = 1 - fx_p
    fx = np.stack([fx_p, fx_n]).reshape((-1, 2))

    ret = get_all_metrics(ys, fxs, fx, ds_dict)

    print(ret)
