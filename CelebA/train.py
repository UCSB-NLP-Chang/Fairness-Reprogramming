import argparse
import os
import time
import sys
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from torch.cuda.amp import autocast, GradScaler
from dataset import CelebAFast as CelebA
from models.model_zoo import *
from models.resnet9 import resnet9
from utils import *
import warnings
warnings.filterwarnings("ignore")

attr_list = ('5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,'
             'Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,'
             'Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,'
             'Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,'
             'Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
             ).split(',')

attr_dict = {}
for i, attr in enumerate(attr_list):
    attr_dict[attr] = i

insufficient_attr_list = '5_o_Clock_Shadow,Goatee,Mustache,Sideburns,Wearing_Necktie'.split(',')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--result-dir', type=str, default='results')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--domain-attrs', type=str, default='Male')
    parser.add_argument('--target-attrs', type=str, default='Blond_Hair')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet20s", "resnet9"])
    parser.add_argument('--evaluate', action="store_true")

    parser.add_argument('--method', '--m', type=str, default="std", choices=['std', 'adv', 'repro', 'rpatch', 'roptim'],
                        help="Method: standard training, adv training, reprogram (vanilla, patch, optimization-based)")

    # ================================ Adv Training ================================ #
    parser.add_argument('--adversary-with-y', action="store_true", default=False,
                        help="True for Equalized Odds, target on [ED_PO1_AcrossZ], "
                             "False for Demographic Parity, target on [ED_FR_AcrossZ].")
    parser.add_argument('--adversary-with-logits', action="store_true", default=False)
    parser.add_argument('--adversary-lr', type=float, default=0.05)
    parser.add_argument('--lmbda', type=float, default=0.5,
                        help="The coefficient of the adversarial loss applied to CE loss")

    # ================================ Reprogramming ================================ #
    parser.add_argument('--reprogram-size', type=int, default=200,
                        help="This parameter has different meanings for different reprogramming methods. "
                             "For vanilla reprogramming method, the size of the resized image."
                             "For reprogram patch, the patch size."
                             "For optimization-based reprogram, the equivalent size of a patch for optimized pixels.")
    parser.add_argument('--trigger-data-num', type=int, default=0,
                        help="How many data do you want to use to train reprogram, default for using the whole training set!")

    args = parser.parse_args()

    args.domain_attrs = args.domain_attrs.split(',')
    args.target_attrs = args.target_attrs.split(',')
    for target_attr in args.target_attrs:
        assert target_attr in attr_list
    for domain_attr in args.domain_attrs:
        assert domain_attr in attr_list

    return args


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)

    # Sanity Check!
    assert args.data_dir is not None
    assert args.method in ["std", "adv"] or args.checkpoint is not None

    # make save path dir
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "csv"), exist_ok=True)
    model_attr_name = args.method + "_" + args.arch + "_" + "_target_"
    for attr in args.target_attrs:
        model_attr_name += str(attr_dict[attr])
        model_attr_name += "_"
    model_attr_name += "domain_"
    for attr in args.domain_attrs:
        model_attr_name += str(attr_dict[attr])
        model_attr_name += "_"
    if args.method in ["adv", "repro", "rpatch", "rpoptim"]:
        model_attr_name += f'lambda{args.lmbda}_'
        model_attr_name += 'y_' if args.adversary_with_y else 'n_'
        if args.method in ["repro", "rpatch", "rpoptim"]:
            model_attr_name += f'size{args.reprogram_size}_'
    if args.trigger_data_num > 0:
        model_attr_name += f'num{args.trigger_data_num}_'
    model_attr_name += f'seed{args.seed}'
    if args.exp_name is not None:
        model_attr_name += f'_{args.exp_name}'

    image_size = 224
    transform_train, transform_test = get_transform(method=args.method,
                                                    image_size=image_size,
                                                    reprogram_size=args.reprogram_size)

    # We will use this a lot.
    use_reprogram = args.method in ["repro", "rpatch", "roptim"]
    use_adv = args.method in ["adv", "repro", "rpatch", "roptim"]

    num_class = 2 ** len(args.target_attrs)
    attr_class = 2 ** len(args.domain_attrs)

    # init model
    if args.arch == "resnet18":
        predictor = resnet18(pretrained=False)
        predictor.fc = nn.Linear(512, num_class)
    elif args.arch == "resnet9":
        predictor = resnet9(num_classes=num_class)
    else:
        predictor = resnet20s(num_class)
    predictor = predictor.to(device)
    p_optim = torch.optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=args.wd)
    p_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(p_optim,
                                                          gamma=0.1,
                                                          milestones=[int(0.8 * args.epochs),
                                                                      int(0.9 * args.epochs)])

    # All the methods except for "std" need adversary
    if use_adv:
        adversary = Adversary(input_dim=num_class, output_dim=attr_class, with_y=args.adversary_with_y,
                              with_logits=args.adversary_with_logits, use_mlp=True)
        adversary = adversary.to(device)
        a_optim = torch.optim.Adam(adversary.parameters(), lr=args.adversary_lr, weight_decay=args.wd)
        a_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(a_optim, gamma=0.1,
                                                              milestones=[int(0.8 * args.epochs),
                                                                          int(0.9 * args.epochs)])
    else:
        adversary = None
        a_optim = None
        a_lr_scheduler = None

    # Initialize reprogrammers
    if use_reprogram:
        reprogram = get_reprogram(method=args.method,
                                  image_size=image_size,
                                  reprogram_size=args.reprogram_size,
                                  device=device)
        r_optim = torch.optim.Adam(reprogram.parameters(), lr=args.lr, weight_decay=args.wd)
        r_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(r_optim,
                                                              gamma=0.1,
                                                              milestones=[int(0.8 * args.epochs),
                                                                          int(0.9 * args.epochs)])
    else:
        # We did so because we need to input reprogram into the eval function
        # We create reprogram here for std/adv mode to simplify the call of eval function
        reprogram = None
        r_optim = None
        r_lr_scheduler = None

    # Load checkpoints
    best_SA = 0.0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        predictor.load_state_dict(checkpoint["predictor"])
        # We use the args.resume to distinguish whether the user want to resume the checkpoint
        # or they just want to load the pretrained models and train the reprogram from scratch.
        if args.resume:
            p_optim.load_state_dict(checkpoint["p_optim"])
            p_lr_scheduler.load_state_dict(checkpoint["p_lr_scheduler"])
            best_SA = checkpoint["best_SA"]
            start_epoch = checkpoint["epoch"]
            if use_adv:
                adversary.load_state_dict(checkpoint["adversary"])
                a_optim.load_state_dict(checkpoint["a_optim"])
                a_lr_scheduler.load_state_dict(checkpoint["a_lr_scheduler"])
            if use_reprogram:
                reprogram.load_state_dict(checkpoint["reprogram"])
                r_optim.load_state_dict(checkpoint["r_optim"])
                r_lr_scheduler.load_state_dict(checkpoint["r_lr_scheduler"])

    test_set = CelebA(args.data_dir, args.target_attrs, args.domain_attrs,
                      img_transform=transform_test, type="test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    if args.evaluate or args.checkpoint is not None:
        print("================= Evaluating on Test Set before Training =================")
        test_result_list, accuracy = evaluation(None, test_loader, predictor, -1, device)
        display_result(accuracy, test_result_list)
        if args.evaluate:
            sys.exit()

    val_set = CelebA(args.data_dir, args.target_attrs, args.domain_attrs,
                     img_transform=transform_test, type="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    if args.trigger_data_num > 0:
        train_set = CelebA(args.data_dir, args.target_attrs, args.domain_attrs,
                           img_transform=transform_train, type="trigger", trigger_data_num=args.trigger_data_num)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    else:
        train_set = CelebA(args.data_dir, args.target_attrs, args.domain_attrs,
                           img_transform=transform_train, type="train", trigger_data_num=args.trigger_data_num)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    no_balanced_result = [["Min_ACC_AcrossZY"], ["ED_FPR_AcrossZ"], ["ED_FNR_AcrossZ"], ["ED_PO1_AcrossZ"],
                          ["Accuracy"]]

    scaler = GradScaler()

    for epoch in range(start_epoch, args.epochs):
        # training
        if args.method == "std":
            predictor.train()
        elif args.method == "adv":
            predictor.train()
            adversary.train()
        else:
            # Freeze the predictor for sure.
            predictor.eval()
            for param in predictor.parameters():
                param.requires_grad = False
            adversary.train()
            reprogram.train()
        end = time.time()
        print(f"======================================= Epoch {epoch} =======================================")
        pbar = tqdm(train_loader, total=len(train_loader), ncols=120)
        total_num = 0
        true_num = 0
        for x, (y, d) in pbar:
            x, y, d = x.to(device), y.to(device), d.to(device)
            y_one_hot = get_one_hot(y, num_class, device)  # one-hot [bs, num_class]
            d_one_hot = get_one_hot(d, attr_class, device)  # one-hot [bs, attr_class]
            p_optim.zero_grad()
            if args.method != "std":
                a_optim.zero_grad()
                if args.method != "adv":
                    r_optim.zero_grad()

            with autocast():
                if reprogram is not None:
                    x = reprogram(x)

                lgt = predictor(x)
                pred_loss = nn.functional.cross_entropy(lgt, y_one_hot)
                if args.method != "std":
                    protect_pred = adversary(lgt, y=y_one_hot if args.adversary_with_y else None)
                    adversary_loss = torch.nn.functional.cross_entropy(protect_pred, d_one_hot)

            if use_adv:
                # The target of the adversary
                working_model = reprogram if use_reprogram else predictor

                scaler.scale(adversary_loss).backward(retain_graph=True)
                adversary_grad = {name: param.grad.clone() for name, param in working_model.named_parameters()}
                scaler.step(a_optim)
                scaler.update()
                if not use_reprogram:
                    p_optim.zero_grad()
                scaler.scale(pred_loss).backward()
                with torch.no_grad():
                    for name, param in working_model.named_parameters():
                        if name in adversary_grad.keys():
                            unit_adversary_grad = adversary_grad[name] / torch.linalg.norm(adversary_grad[name])
                            param.grad -= (param.grad * unit_adversary_grad).sum() * unit_adversary_grad
                            param.grad -= args.lmbda * adversary_grad[name]
                    del adversary_grad
                if use_reprogram:
                    scaler.step(r_optim)
                    scaler.update()
                else:
                    scaler.step(p_optim)
                    scaler.update()
            else:
                scaler.scale(pred_loss).backward()
                scaler.step(p_optim)
                scaler.update()

            # results for this batch
            total_num += y.size(0)
            true_num += (lgt.argmax(1) == y.view(-1)).type(torch.float).sum().detach().cpu().item()
            acc = true_num * 1.0 / total_num
            pbar.set_description(f"Training Epoch {epoch} Acc {100 * acc:.2f}%")
        pbar.set_description(f"Training Epoch {epoch} Acc {100 * true_num / total_num:.2f}%")

        if args.method == "std":
            p_lr_scheduler.step()
        elif args.method == "adv":
            p_lr_scheduler.step()
            a_lr_scheduler.step()
        else:
            a_lr_scheduler.step()
            r_lr_scheduler.step()

        # evaluating
        print("================= Evaluating on Validation Set =================")
        res, accuracy = evaluation(reprogram, val_loader, predictor, epoch, device)

        load_result(res, no_balanced_result, accuracy)
        display_result(accuracy, res)
        write_csv_rows(os.path.join(os.path.join(args.result_dir, "csv"), f'{model_attr_name}.csv'),
                       no_balanced_result)
        if args.method == "std":
            # For standard training, we select the one with highest accuracy
            metric = accuracy
        else:
            # For others, we select the one with the highest Min_ACC_AcrossZY.
            metric = res["Min_ACC_AcrossZY"]
        if metric > best_SA:
            print("+++++++++++ Find New Best Min ACC +++++++++++")
            best_SA = metric
            cp = {"predictor": predictor.state_dict(),
                  "p_optim": p_optim.state_dict(),
                  "p_lr_scheduler": p_lr_scheduler.state_dict(),
                  "epoch": epoch,
                  "best_SA": best_SA
                  }
            if args.method != "std":
                cp["adversary"] = adversary.state_dict()
                cp["a_optim"] = a_optim.state_dict()
                cp["a_lr_scheduler"] = a_lr_scheduler.state_dict()
                if args.method != "adv":
                    cp["reprogram"] = reprogram.state_dict()
                    cp["r_optim"] = r_optim.state_dict()
                    cp["r_lr_scheduler"] = r_lr_scheduler.state_dict()
            torch.save(cp,
                       os.path.join(os.path.join(args.result_dir, "checkpoints"), f'{model_attr_name}_best.pth.tar'))
        print("================= Test Set =================")
        test_result_list, accuracy = evaluation(reprogram, test_loader, predictor, epoch, device)

        display_result(accuracy, test_result_list)

        print(f"Time Consumption for one epoch is {time.time() - end}s")


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
