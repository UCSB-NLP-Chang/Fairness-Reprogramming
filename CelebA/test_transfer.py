import argparse
from torch.cuda.amp import autocast
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18

from dataset import CelebA
from metric import *
from utils import *
from models.model_zoo import *
from program import AdvProgram, PatchProgram

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
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18"])
    parser.add_argument('--data-path', type=str, default=None, required=True)
    parser.add_argument('--save-path', type=str, default='results')
    parser.add_argument('--load-path', type=str, default=None, required=True)
    parser.add_argument('--load-trigger', type=str, default=None)
    parser.add_argument('--in-size', type=int, default=170)
    parser.add_argument('--trigger-type', type=str, default="normal", choices=["normal", "patch"])
    parser.add_argument('--patch-size', type=int, default=40)
    parser.add_argument('--patch-loc', type=int, default=0, choices=[0, 1, 2, 3],
                        help="left upper corner, right upper corner, left bottom corner, right bottom corner")
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--domain-attrs', type=str, default='Male')
    parser.add_argument('--target-attrs', type=str, default='Blond_Hair')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--display', action="store_true", default=False)

    args = parser.parse_args()

    args.domain_attrs = args.domain_attrs.split(',')
    assert args.target_attrs in attr_list
    args.target_attrs = [args.target_attrs]

    return args


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_class = 2

    # init model
    if args.arch == "resnet18":
        predictor = resnet18(pretrained=False)
        predictor.fc = nn.Linear(512, num_class)
    else:
        predictor = resnet20s(num_class)
    predictor = predictor.to(device)
    predictor.load_state_dict(torch.load(args.load_path, map_location=device))

    print("===============================Start Evaluate Standard Model ==================================")
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    test_set = CelebA(args.data_path, args.target_attrs, args.domain_attrs,
                      img_transform=transform_test, type="test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # evaluating
    predictor.eval()
    pbar = tqdm(test_loader, total=len(test_loader), ncols=120, desc="Testing")
    fxs = []
    fxs_prob = []
    ys = []
    ds = []
    test_total_num = 0
    test_true_num = 0
    for x, y, d in pbar:
        ys.append(y)
        ds.append(d)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad(), autocast():
            lgt = predictor(x)
            fxs_prob.append(lgt)
        test_total_num += y.shape[0]
        pred = lgt.argmax(1)  # [bs]
        fxs.append(pred)
        test_true_num += (pred == y.view(-1)).type(torch.float).sum().detach().cpu().item()
        acc = test_true_num * 1.0 / test_total_num
        pbar.set_description(f"Test Acc {100 * acc:.2f}%")
    pbar.set_description(f"Test Acc {100 * test_true_num / test_total_num:.2f}%")

    ys, ds = torch.cat(ys).view(-1).cpu().numpy(), torch.cat(ds).view(-1).cpu().numpy()
    ds_dict = {"Male": ds, "Female": 1 - ds}
    fxs = torch.cat(fxs).view(-1).detach().cpu().numpy()
    fxs_prob = torch.cat(fxs_prob, dim=0).detach().cpu().numpy()

    # So that there will not be nan to the get_all_metrics function
    fxs_prob = np.clip(fxs_prob, a_min=-100000.0, a_max=100000.0)

    ret_no_class_balance = get_all_metrics(y_true=ys, y_pred=fxs, y_prob=fxs_prob, z=ds_dict,
                                           use_class_balance=False)
    accuracy = test_true_num / test_total_num

    print(f"test acc: {accuracy:.4f}")
    print(f"Min_ACC_AcrossZY: {ret_no_class_balance['Min_ACC_AcrossZY']: .4f}")
    print(f"ED_FPR_AcrossZ: {ret_no_class_balance['ED_FPR_AcrossZ']: .4f}")
    print(f"ED_FNR_AcrossZ: {ret_no_class_balance['ED_FNR_AcrossZ']: .4f}")
    print(
        f"ED_FR_AcrossZ: {(ret_no_class_balance['ED_FPR_AcrossZ'] + ret_no_class_balance['ED_FNR_AcrossZ']): .4f}")
    print(f"ED_PO1_AcrossZ: {ret_no_class_balance['ED_PO1_AcrossZ']: .4f}")

    print("===============================Start Evaluating Reprogramming ==================================")
    # Reprogram configuration
    if args.trigger_type == "normal":

        in_size = (args.in_size, args.in_size)
        out_size = (224, 224)
        mask_size = out_size

        l_pad = int((out_size[0] - in_size[0] + 1) / 2)
        r_pad = int((out_size[0] - in_size[0]) / 2)

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(in_size[0]),
            transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
            transforms.ToTensor(),
        ])

        # init reprogramme
        adv_program = AdvProgram(in_size, out_size, mask_size, device=device)

    else:
        out_size = (224, 224)
        patch_size = (args.patch_size, args.patch_size)

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        # init reprogramme
        adv_program = PatchProgram(patch_size=patch_size, out_size=out_size, loc=args.patch_loc, device=device)

    if args.load_trigger is not None:
        adv_program.load_state_dict(torch.load(args.load_trigger, map_location=device))

    test_set = CelebA(args.data_path, args.target_attrs, args.domain_attrs,
                      img_transform=transform_test, type="test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # evaluating
    adv_program.eval()
    pbar = tqdm(test_loader, total=len(test_loader), ncols=120, desc="Testing")
    fxs = []
    fxs_prob = []
    y_all = []
    d_all = []
    test_total_num = 0
    test_true_num = 0
    for x, y, d in pbar:
        y_all.append(y)
        d_all.append(d)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = adv_program(x)

            if args.display:
                plt.imshow(x[0].permute(1, 2, 0))
                plt.title(f"{args.trigger_type} Trigger, Size {args.patch_size}")
                plt.show()
                return

            lgt = predictor(x)
            fxs_prob.append(lgt)
        test_total_num += y.shape[0]
        pred = lgt.argmax(1)  # [bs]
        fxs.append(pred)
        test_true_num += (pred == y.view(-1)).type(torch.float).sum().detach().cpu().item()
        acc = test_true_num * 1.0 / test_total_num
        pbar.set_description(f"Test Acc {100 * acc:.2f}%")
    pbar.set_description(f"Test Acc {100 * test_true_num / test_total_num:.2f}%")

    y_all, d_all = torch.cat(y_all).view(-1).cpu().numpy(), torch.cat(d_all).view(-1).cpu().numpy()
    ds_dict = {"Male": d_all, "Female": 1 - d_all}
    fxs = torch.cat(fxs).view(-1).detach().cpu().numpy()
    fxs_prob = torch.cat(fxs_prob, dim=0).detach().cpu().numpy()

    ret_no_class_balance = get_all_metrics(y_true=y_all, y_pred=fxs, y_prob=fxs_prob, z=ds_dict,
                                           use_class_balance=False)

    accuracy = test_true_num / test_total_num
    print(f"test acc: {accuracy:.4f}")
    print(f"Min_ACC_AcrossZY: {ret_no_class_balance['Min_ACC_AcrossZY']: .4f}")
    print(f"ED_FPR_AcrossZ: {ret_no_class_balance['ED_FPR_AcrossZ']: .4f}")
    print(f"ED_FNR_AcrossZ: {ret_no_class_balance['ED_FNR_AcrossZ']: .4f}")
    print(
        f"ED_FR_AcrossZ: {(ret_no_class_balance['ED_FPR_AcrossZ'] + ret_no_class_balance['ED_FNR_AcrossZ']): .4f}")
    print(f"ED_PO1_AcrossZ: {ret_no_class_balance['ED_PO1_AcrossZ']: .4f}")


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
