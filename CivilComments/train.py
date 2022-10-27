import copy
import math
import os
import sys

import torch
import torch.nn as nn
import transformers
from torch.nn import DataParallel
from tqdm import tqdm

from eval import evaluate
from models import Adversary4Z
from utils import set_logger_config_recorder_timer_seed, torch_save, set_dataset_device_model, _to_device, \
    dir_all_models
from zarth_utils.general_utils import makedir_if_not_exist
from zarth_utils.result_recorder import load_result


class FastMMD:
    """ Fast Maximum Mean Discrepancy approximated using the random kitchen sinks method.
    """

    def __init__(self, device, in_features, out_features=500, gamma=1.0):
        # W sampled from normal
        self.w_rand = torch.randn((in_features, out_features)).to(device)
        # b sampled from uniform
        self.b_rand = torch.zeros((out_features,)).uniform_(0, 2 * math.pi).to(device)
        self.scale_a = math.sqrt(2 / out_features)
        self.scale_b = math.sqrt(2 / gamma)

    def __call__(self, a, b, *args, **kwargs):
        phi_a = self._phi(a, self.w_rand, self.b_rand)
        phi_b = self._phi(b, self.w_rand, self.b_rand)

        mmd = torch.norm(phi_a.mean(0) - phi_b.mean(dim=0), 2)
        return mmd

    def _phi(self, x, w_rand, b_rand):
        out = self.scale_a * (self.scale_b * (x @ w_rand + b_rand)).cos()
        return out


def train_loop(model, train_dl, device, optimizer, scheduler, num_label, adversary_loss_weight=0.0, adversary=None,
               adversary_with_y=None, adversary_optimizer=None, adversary_scheduler=None, use_trigger=None,
               num_batch_per_epoch=None, pseudo_model=None, gradient_clipping=-1.0, ce_loss_weight=1.0,
               multi_label_task=False, mmd_loss_weight=0.0, use_adversary_projection=True):
    """
    Train the model for one epoch.
    :param model: the model
    :type model: torch.nn.Module
    :param train_dl: the trianing set
    :type train_dl: torch.utils.data.DataLoader
    :param device: the device
    :type device: torch.device
    :param optimizer: the optimizer
    :type optimizer: torch.optim.Optimizer
    :param scheduler: the learning rate scheduler
    :type scheduler: torch.optim.lr_scheduler.Scheduler
    :param num_label: the number of labels
    :param adversary_loss_weight: the weight for the adversary loss
    :param adversary: the adversary model
    :param adversary_with_y: whether the input to the adversary model contains y
    :param adversary_optimizer: the optimizer for the adversary model
    :param adversary_scheduler: the scheduler for the adversary model
    :param use_trigger: whether use un-adversarial triggers
    :param num_batch_per_epoch: batch numbers per epoch
    :param pseudo_model: model to generate pseudo labels
    :param gradient_clipping: gradient_clipping value
    :param ce_loss_weight: weight of ce loss
    :param multi_label_task: must use bce with whatever num label, suitable for multi-label setting
    :param mmd_loss_weight: weight of mmd loss
    :param use_adversary_projection: whether remove ce gradient projection on adversary grad
    """
    model.train()
    acc_all, loss_all, adv_loss_all, len_dl = 0, 0, 0, 0

    for i, (idx, x, y, z) in enumerate(tqdm(train_dl, file=sys.stdout)):
        if i == num_batch_per_epoch:
            break

        x, y, z = _to_device(x, device), _to_device(y, device), _to_device(z, device)
        if pseudo_model is not None:
            y = pseudo_model(*x) if type(x) in [tuple, list] else pseudo_model(x)[-1].view(-1)

        o, prob, pred = model(*x) if type(x) in [tuple, list] else model(x)

        adversary_grad, adversary_loss = None, torch.tensor(0)
        if adversary is not None and adversary_loss_weight > 0:
            adversary_optimizer.zero_grad()
            optimizer.zero_grad()
            o_adversary = adversary(o, y=y if adversary_with_y else None)
            if z.shape[1] == 2 and o_adversary.shape[1] == 1:
                z = z[:, 0].view(-1, 1)
            adversary_loss = nn.functional.binary_cross_entropy_with_logits(o_adversary, z.float())
            adversary_loss.backward(retain_graph=True)
            if use_trigger:
                adversary_grad = {
                    name: param.grad.clone() for name, param in model.named_parameters()
                    if "trigger" in name and param.requires_grad and param.grad is not None
                }
            else:
                adversary_grad = {
                    name: param.grad.clone() for name, param in model.named_parameters()
                    if param.requires_grad and param.grad is not None
                }
            adversary_optimizer.step()

        if multi_label_task or num_label == 1:
            loss = nn.functional.binary_cross_entropy_with_logits(o, y.view(o.shape[0], -1).float()) * ce_loss_weight
        else:
            loss = nn.functional.cross_entropy(o, y) * ce_loss_weight

        if math.fabs(mmd_loss_weight) > 1e-6:
            mmd = FastMMD(device=device, in_features=o.shape[1])
            for j in range(z.shape[1]):
                if z[:, j].sum() == 0:
                    continue
                loss += mmd_loss_weight * mmd(o, o[z[:, j] == 1])
        else:
            mmd = None

        optimizer.zero_grad()
        loss.backward()

        if mmd is not None:
            del mmd

        if adversary is not None and adversary_loss_weight > 0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in adversary_grad.keys():
                        if use_adversary_projection:
                            unit_adversary_grad = adversary_grad[name] / torch.linalg.norm(adversary_grad[name])
                            param.grad -= (param.grad * unit_adversary_grad).sum() * unit_adversary_grad
                        param.grad -= adversary_loss_weight * adversary_grad[name]

        if gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()

        acc_all += (pred.view(-1) == y).type(torch.float).sum().item()
        loss_all += loss.item() * len(x)
        adv_loss_all += adversary_loss.item() * len(x)
        len_dl += len(y)

        if scheduler is not None:
            scheduler.step()
        if adversary_scheduler is not None:
            adversary_scheduler.step()

        if type(model) == nn.DataParallel:
            model.module.project_trigger_word_selector()
        else:
            model.project_trigger_word_selector()

    acc_all = acc_all / len_dl * 100.0 / (num_label if multi_label_task else 1)
    loss_all /= len_dl * (num_label if multi_label_task else 1)
    adv_loss_all /= len_dl * (num_label if multi_label_task else 1)

    return acc_all, loss_all, adv_loss_all


def set_optimizer_scheduler(optimizer_name, scheduler_name, model, lr, weight_decay, momentum, total_steps=None,
                            only_optimize_trigger=False):
    if only_optimize_trigger:
        parameters_to_be_optimized = [(n, p) for n, p in model.named_parameters() if "trigger" in n]
    else:
        parameters_to_be_optimized = model.named_parameters()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in parameters_to_be_optimized if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in parameters_to_be_optimized if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=lr,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=lr,
            momentum=momentum
        )
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(
            optimizer_grouped_parameters,
            lr=lr,
        )
    elif optimizer_name == "adamw":
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            eps=1e-8
        )
    else:
        raise NotImplementedError

    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.96)
    elif scheduler_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(total_steps * 0.5), int(total_steps * 0.75)],
            gamma=0.1
        )
    elif scheduler_name == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=total_steps
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def main():
    ###################################################################################
    # Setting up the config, logger, result recorder and timer
    ###################################################################################
    path_config = os.path.join(os.getcwd(), "configs", "config_train.json")
    logger, config, res_recorder, timer, dir_model = set_logger_config_recorder_timer_seed(path_config)

    ###################################################################################
    # Loading the dataset, device and model
    ###################################################################################
    train_dl, val_dl_dict, test_dl_dict, num_label, num_group, device, model = set_dataset_device_model(config)
    logger.info(model)
    makedir_if_not_exist(dir_model)

    ###################################################################################
    # Setting up optimizer and scheduler
    ###################################################################################

    num_batch_per_epoch = config["num_batch_per_epoch"]
    if not (0 < num_batch_per_epoch < len(train_dl)):
        num_batch_per_epoch = len(train_dl)
    optimizer, scheduler = set_optimizer_scheduler(
        optimizer_name=config["optimizer"], scheduler_name=config["lr_scheduler"], model=model, lr=config["lr"],
        weight_decay=config["weight_decay"], momentum=config["momentum"],
        total_steps=config["num_epoch"] * num_batch_per_epoch, only_optimize_trigger=config["only_optimize_trigger"]
    )

    ###################################################################################
    # Setting up adversary
    ###################################################################################
    adversary, adversary_optimizer, adversary_scheduler = None, None, None
    if config["adversary_loss_weight"] > 0:
        adversary = Adversary4Z(input_dim=num_label, output_dim=num_group, with_y=config["adversary_with_y"],
                                with_logits=config["adversary_with_logits"], use_mlp=config["adversary_use_mlp"],
                                with_logits_y=config["adversary_with_logits_y"],
                                with_single_y=config["adversary_with_single_y"])
        if config["use_data_parallel"]:
            adversary = DataParallel(adversary)
        adversary = adversary.to(device)
        logger.info(adversary)

        adversary_optimizer, adversary_scheduler = set_optimizer_scheduler(
            optimizer_name=config["adversary_optimizer"], scheduler_name=config["adversary_lr_scheduler"],
            model=adversary, lr=config["adversary_lr"], weight_decay=config["adversary_weight_decay"],
            momentum=config["adversary_momentum"], total_steps=config["num_epoch"] * num_batch_per_epoch
        )

    ###################################################################################
    # Train the model
    ###################################################################################
    validation_standard = {"max": 1, "min": -1}[config["validation_standard"]]
    best_score, best_epoch, no_improvement = -validation_standard * math.inf, 0, 0

    if config["dir_pretrain_model"] != "":
        model.load_state_dict(
            torch.load(os.path.join(dir_all_models, config["dir_pretrain_model"], "best_model.pth")),
            strict=False
        )
        pretrain_train_record, pretrain_finished = load_result(
            os.path.join(dir_all_models, config["dir_pretrain_model"], "train.result"))
        assert pretrain_finished
        res_recorder.update({("pretraining-%s" % k): v for k, v in pretrain_train_record.items()})

    pseudo_model = None
    if config["use_pseudo_label"]:
        assert config["dir_pretrain_model"] != ""
        pseudo_model = copy.deepcopy(model)
        for param in pseudo_model.parameters():
            param.requires_grad = False
        pseudo_model.eval()

    # Continue training from config["start_epoch"] + 1
    if config["start_epoch"] != 0:
        ckpt = torch.load(os.path.join(dir_model, 'epoch%d_checkpoint.pth' % config["start_epoch"]))
        best_score, best_epoch, no_improvement = ckpt["best_score"], ckpt["best_epoch"], ckpt["no_improvement"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        model.load_state_dict(
            torch.load(os.path.join(dir_model, 'epoch%d_model_weights.pth' % config["start_epoch"]))
        )

    # Evaluate the model before training
    logger.info("Epoch %d:" % config["start_epoch"])
    evaluate(model, device, logger, res_recorder, val_dl_dict, epoch=config["start_epoch"],
             use_sample4eval=config["sample4eval"])
    if config["num_trigger_word"] != 0 and not config["trigger_on_embedding"]:
        trigger = model.get_trigger_words()
        res_recorder.add_with_logging(key="epoch_%d-trigger" % config["start_epoch"], value=trigger,
                                      msg="\t\tTrigger is: %s")

    # Start training
    for epoch in range(config["start_epoch"] + 1, config["num_epoch"] + 1):
        # Train the model for one epoch
        timer.start()
        acc_train, loss_train, adv_loss_train = train_loop(
            model=model,
            train_dl=train_dl,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            num_label=num_label,
            adversary_loss_weight=config["adversary_loss_weight"],
            adversary=adversary,
            adversary_optimizer=adversary_optimizer,
            adversary_scheduler=adversary_scheduler,
            adversary_with_y=config["adversary_with_y"],
            use_trigger=(config["num_trigger_word"] > 0),
            num_batch_per_epoch=config["num_batch_per_epoch"],
            pseudo_model=pseudo_model,
            gradient_clipping=config["gradient_clipping"],
            ce_loss_weight=config["ce_loss_weight"],
            mmd_loss_weight=config["mmd_loss_weight"],
            use_adversary_projection=config["use_adversary_projection"]
        )
        time_train = timer.get_last_duration()

        # Record and present the results
        logger.info("Epoch %d:" % epoch)
        logger.info("\tTraining Set:")
        res_recorder.add_with_logging("epoch_%d-acc_train" % epoch, float(acc_train), "\t\tAccuracy: %.4lf")
        res_recorder.add_with_logging("epoch_%d-loss_train" % epoch, float(loss_train), "\t\tLoss: %.4lf")
        res_recorder.add_with_logging("epoch_%d-adv_loss_train" % epoch, float(adv_loss_train), "\t\tAdv Loss: %.4lf")
        res_recorder.add_with_logging("epoch_%d-time_train" % epoch, float(time_train), "\t\tTime: %.2lfs")

        # Evaluate the model on the validation set
        results_val = evaluate(model, device, logger, res_recorder, val_dl_dict, epoch=epoch,
                               use_sample4eval=config["sample4eval"])
        if config["num_trigger_word"] != 0 and not config["trigger_on_embedding"]:
            trigger = model.get_trigger_words()
            res_recorder.add_with_logging(key="epoch_%d-trigger" % epoch, value=trigger, msg="\t\tTrigger is: %s")

        # Save the model
        if epoch % config["saving_frequency"] == 0 and not config["only_save_best_model"]:
            checkpoint = {"best_score": best_score, "best_epoch": best_epoch, "no_improvement": no_improvement,
                          "optimizer_state_dict": optimizer.state_dict()}
            if scheduler is not None:
                checkpoint.update({"scheduler_state_dict": scheduler.state_dict()})
            torch_save(model.state_dict(), os.path.join(dir_model, 'epoch%d_model_weights.pth' % epoch))
            torch_save(checkpoint, os.path.join(dir_model, 'epoch%d_checkpoint.pth' % epoch))

        if epoch <= config["num_warmup_epoch"]:
            continue

        val_dl_name = "val_standard_sample4eval" if config["sample4eval"] else "val_standard"
        score = results_val[val_dl_name][config["validation_metric"]]
        if score * validation_standard > best_score * validation_standard:
            logger.info("\tUpdating the best model from %.4lf to %.4lf!" % (best_score, score))
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), os.path.join(dir_model, 'best_model.pth'))
            checkpoint = {"best_score": best_score, "best_epoch": best_epoch,
                          "optimizer_state_dict": optimizer.state_dict()}
            if scheduler is not None:
                checkpoint.update({"scheduler_state_dict": scheduler.state_dict()})
            torch.save(checkpoint, os.path.join(dir_model, 'best_checkpoint.pth'))
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == config["max_no_improvement"]:
                break

    res_recorder.add_with_logging(key="best_epoch", value=best_epoch, msg="Best Epoch: %d")
    res_recorder.add_with_logging(key="best_score", value=best_score, msg="Best Score: %.4lf")
    if config["num_trigger_word"] != 0 and not config["trigger_on_embedding"]:
        model.load_state_dict(torch.load(os.path.join(dir_model, 'best_model.pth')))
        trigger = model.get_trigger_words()
        res_recorder.add_with_logging(key="trigger", value=trigger, msg="Trigger is: %s")
    logger.info("End with Total Time Use: %.2lf!" % timer.get_cumulative_duration())
    res_recorder.end_recording()


if __name__ == '__main__':
    main()
