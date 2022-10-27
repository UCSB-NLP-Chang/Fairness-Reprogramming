import numpy as np
import pandas as pd
from zarth_utils.result_recorder import collect_results, remove_duplicate

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

data = collect_results("models")
data = data[[c for c in data.columns if not (c.startswith("epoch_") or c.startswith("pretraining-epoch"))]]
data = remove_duplicate(data)

train_data = data[data["phase"] == "train"].dropna(axis=1, how="all")
eval_data = data[data["phase"] == "eval"].dropna(axis=1, how="all")
data = train_data.merge(eval_data, on=["exp_name"], how="inner", suffixes=["", "_eval"])


def filter_acc_score(t):
    if not pd.isna(t["acc_test_standard"]):
        return t["acc_test_standard"]
    else:
        return t["acc_test_standard_sample4eval"]


def filter_fairness_score(t):
    if t["validation_metric"] == "ED_PO1_AcrossZ":
        if not pd.isna(t["ed_po1_acrossz_test_standard"]):
            return t["ed_po1_acrossz_test_standard"]
        else:
            return t["ed_po1_acrossz_test_standard_sample4eval"]

    else:
        if not pd.isna(t["ed_fr_acrossz_test_standard"]):
            return t["ed_fr_acrossz_test_standard"]
        else:
            return t["ed_fr_acrossz_test_standard_sample4eval"]


data["accuracy"] = data.apply(filter_acc_score, axis=1)
data["fairness_score"] = data.apply(filter_fairness_score, axis=1)

columns_group_on = ["validation_metric", "adversary_with_y", "use_training2", "only_optimize_trigger",
                    "trigger_on_embedding", "num_warmup_epoch", "num_trigger_word", "sample4eval",
                    "adversary_loss_weight"]
columns_group_on = [c for c in columns_group_on if not data[c].hasnans and len(np.unique(data[c].values)) > 1]
metrics_of_interest = ["accuracy", "fairness_score"]
for c in metrics_of_interest:
    data[c] = data[c].apply(lambda t: np.array(t).astype(np.float32))


def show_result(res):
    res = res[columns_group_on + metrics_of_interest]
    _columns_group_on = [c for c in columns_group_on if not res[c].hasnans and len(np.unique(res[c].values)) > 1]
    res = res[(res["adversary_loss_weight"] == 0.0) | (res["num_warmup_epoch"] == 0)]
    # print(res.groupby(_columns_group_on).mean())

    res2 = dict()
    res2["Base"] = res[
        (res["only_optimize_trigger"] == False) &
        (res["use_training2"] == False) &
        (res["adversary_loss_weight"] == 0.0) &
        (res["num_warmup_epoch"] == 8)
        ].groupby(_columns_group_on).mean()
    res2["AdvIn"] = res[
        (res["only_optimize_trigger"] == False) &
        (res["use_training2"] == False) &
        (res["adversary_loss_weight"] >= 0.0) &
        (res["num_warmup_epoch"] == 0)].groupby(_columns_group_on).mean()
    res2["AdvPost"] = res[
        (res["only_optimize_trigger"] == False) &
        (res["use_training2"] == True) &
        (res["adversary_loss_weight"] >= 0.0) &
        (res["num_warmup_epoch"] == 0)].groupby(_columns_group_on).mean()
    res2["Soft"] = res[
        (res["only_optimize_trigger"] == True) &
        (res["sample4eval"] == True) &
        (res["num_warmup_epoch"] == 0)].groupby(_columns_group_on).mean()
    res2["Hard"] = res[
        (res["only_optimize_trigger"] == True) &
        (res["sample4eval"] == False) &
        (res["num_warmup_epoch"] == 0)].groupby(_columns_group_on).mean()
    for k in res2:
        print("=======================================================================================================")
        print("%s Results:" % k)
        print(res2[k])
        print()


print("Demographic Parity:")
filters_dp = (data["validation_metric"] == "ED_PO1_AcrossZ")
dp_data = data[filters_dp]
show_result(dp_data)

print()
print()
print("-------------------------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------------------------")
print()
print()

print("Equalized Odds:")
filters_eo = (data["validation_metric"] == "ED_FR_AcrossZ")
fr_data = data[filters_eo]
show_result(fr_data)
