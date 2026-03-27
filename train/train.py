import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
# ==========================================================
# 修改點 1: 引入 build_model (取代原本的 compressai.zoo import image_models)
# ==========================================================
from compressai.models.cic import build_model 
import wandb


class RateDistortionLoss(nn.Module):
    """
    Rate-Distortion Loss with ROI (Region of Interest) weighting.
    """

    def __init__(self, lmbda=1e-2, roi_factor=0.1):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # Pixel-wise for weighting
        self.mse_raw = nn.MSELoss()              # For fair PSNR calculation
        self.lmbda = lmbda
        self.roi_factor = roi_factor

    def generate_water_mask(self, images):
        """ Generate water body mask using RGB + saturation heuristics. """
        images = torch.clamp(images, 0, 1)
        
        r = images[:, 0, :, :]
        g = images[:, 1, :, :]
        b = images[:, 2, :, :]
        
        # RGB conditions
        cond1 = b > (r * 1.1)  # Blue dominant over red
        cond2 = b > g          # Blue dominant over green
        cond3 = b < 0.9        # Not too bright (clouds)
        
        # Saturation condition
        max_rgb = torch.max(torch.max(r, g), b)
        min_rgb = torch.min(torch.min(r, g), b)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        cond4 = saturation < 0.3
        
        is_water = cond1 & cond2 & cond3 & cond4
        
        # Create weight map (land = 1.0, water = roi_factor)
        weight_map = torch.ones_like(r)
        weight_map[is_water] = self.roi_factor
        
        return weight_map.unsqueeze(1)  # (B, 1, H, W)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # Bpp loss
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        # Generate water mask from target image
        roi_weights = self.generate_water_mask(target)

        # Weighted MSE (for training)
        mse_pixel_wise = self.mse(output["x_hat"], target)
        weighted_mse = mse_pixel_wise * roi_weights
        out["mse_loss"] = weighted_mse.mean()

        # Raw MSE (for fair PSNR calculation)
        out["mse_loss_raw"] = self.mse_raw(output["x_hat"], target)

        # Total loss
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    # ==========================================================
    # 修改點 2: 資料夾命名加入 N 和 M，避免不同架構的權重混在一起
    # ==========================================================
    base_dir = f'./pretrained/{args.model}_N{args.N}_M{args.M}/q{args.quality_level}/L{args.lmbda:.4f}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer."""
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    
    # Check for parameter overlap (sanity check)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        import torchvision.transforms.functional as TF

        def psnr(mse_val):
            return 10 * math.log10(1.0 / (mse_val + 1e-10))

        if i * len(d) % 5000 == 0 or i == 0:
            mse_value = out_criterion["mse_loss"].item()
            mse_raw_value = out_criterion["mse_loss_raw"].item()
            bpp_value = out_criterion["bpp_loss"].item()
            aux_value = aux_loss.item()
            total_loss = out_criterion["loss"].item()

            psnr_val = psnr(mse_raw_value)

            input_img = d[0].detach().cpu()
            recon_img = out_net["x_hat"][0].detach().cpu()
            input_img = torch.clamp(input_img, 0, 1)
            recon_img = torch.clamp(recon_img, 0, 1)

            logging.info(
                f"[{i * len(d)}/{len(train_dataloader.dataset)}] | "
                f"Loss: {total_loss:.3f} | "
                f"MSE(w): {mse_value:.5f} | "
                f"MSE(raw): {mse_raw_value:.5f} | "
                f"Bpp: {bpp_value:.4f} | "
                f"Aux: {aux_value:.2f} | "
                f"PSNR: {psnr_val:.2f}"
            )

            wandb.log({
                "step": i + epoch * len(train_dataloader),
                "loss": total_loss,
                "mse_loss_weighted": mse_value,
                "mse_loss_raw": mse_raw_value,
                "bpp_loss": bpp_value,
                "aux_loss": aux_value,
                "psnr": psnr_val,
                "original vs recon": [
                    wandb.Image(TF.to_pil_image(input_img), caption="Input"),
                    wandb.Image(TF.to_pil_image(recon_img), caption="Reconstructed"),
                ]
            })


def eval_epoch(epoch, dataloader, model, criterion):
    """Evaluate on validation/test set."""
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    mse_loss_raw = AverageMeter()
    aux_loss = AverageMeter()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for d in dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            mse_loss_raw.update(out_criterion["mse_loss_raw"])

            mse_val = out_criterion["mse_loss_raw"].item()
            psnr_val = 10 * math.log10(1.0 / mse_val) if mse_val > 0 else float('inf')
            psnr_meter.update(psnr_val)

    log_prefix = "Test" if isinstance(epoch, str) else "Val"

    logging.info(
        f"{log_prefix} epoch {epoch}: "
        f"Loss: {loss.avg:.3f} | "
        f"MSE(w): {mse_loss.avg:.5f} | "
        f"MSE(raw): {mse_loss_raw.avg:.5f} | "
        f"PSNR: {psnr_meter.avg:.2f} dB | "
        f"Bpp: {bpp_loss.avg:.4f} | "
        f"Aux: {aux_loss.avg:.2f}\n"
    )

    wandb.log({
        f"{log_prefix.lower()}_loss": loss.avg,
        f"{log_prefix.lower()}_mse_weighted": mse_loss.avg,
        f"{log_prefix.lower()}_mse_raw": mse_loss_raw.avg,
        f"{log_prefix.lower()}_psnr": psnr_meter.avg,
        f"{log_prefix.lower()}_bpp": bpp_loss.avg,
    })

    return loss.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir + filename)
    if is_best:
        shutil.copyfile(base_dir + filename, base_dir + "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="CIC Mean-Scale Training Script")
    parser.add_argument(
        "-m", "--model",
        default="cic",
        help="Model architecture (default: %(default)s)",
    )
    # ==========================================================
    # 修改點 3: 加入 N 和 M 參數，允許控制 Teacher 模型大小
    # ==========================================================
    parser.add_argument("--N", type=int, default=128, help="Hidden channels N (default: 128)")
    parser.add_argument("--M", type=int, default=192, help="Latent channels M (default: 192)")
    
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset path"
    )
    parser.add_argument(
        "-e", "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr", "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        default=16,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-q", "--quality-level",
        type=int,
        default=3,
        help="Quality level (Used for folder naming & lambda default)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=int, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="Gradient clipping max norm (default: %(default)s)",
    )
    parser.add_argument(
        '--name',
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    
    parser.add_argument(
        "--roi-factor",
        type=float,
        default=1.0,
        help="Weight for water regions in ROI loss (default: %(default)s)",
    )
    
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    wandb.init(
        project="cic-mean-scale-training",
        name=args.name,
        config=vars(args)
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    logging.info("Model: CIC Mean-Scale Hyperprior + ROI Loss")
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(args.patch_size),
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    val_dataset = ImageFolder(args.dataset, split="val", transform=test_transforms)

    # Test dataset is optional
    test_dir = os.path.join(args.dataset, "test")
    if os.path.isdir(test_dir):
        test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
        logging.info(f"Found {len(test_dataset)} test images")
    else:
        test_dataset = None
        logging.info("No test folder found, skipping test dataset.")

    logging.info(f"Found {len(train_dataset)} training images")
    logging.info(f"Found {len(val_dataset)} validation images")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )
    else:
        test_dataloader = None

    # ==========================================================
    # 修改點 4: 使用 build_model 初始化模型 (不再使用 image_models[key](quality))
    # ==========================================================
    logging.info(f"Building Model: {args.model} with N={args.N}, M={args.M}")
    net = build_model(args.model, N=args.N, M=args.M)
    net = net.to(device)
    
    # Log model architecture info
    # 注意: 如果 M 值改變，這裡的檢查可能需要調整，但基本概念不變
    try:
        logging.info(f"h_s output channels: {net.h_s[-1].out_channels} (should be {net.M * 2} for Mean-Scale)")
    except Exception:
        pass # 避免因為網路結構不同而 crash

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 350], gamma=0.1)
    
    # Initialize loss with ROI factor
    criterion = RateDistortionLoss(lmbda=args.lmbda, roi_factor=args.roi_factor)
    logging.info(f"ROI Factor: {args.roi_factor} (water regions weighted at {args.roi_factor * 100}%)")

    last_epoch = 0
    if args.checkpoint:
        logging.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info(f'====== Epoch {epoch} ======')
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )

        val_loss = eval_epoch(epoch, val_dataloader, net, criterion)
        lr_scheduler.step()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": val_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "N": args.N,
                    "M": args.M,
                },
                is_best,
                base_dir
            )

    logging.info("Training finished.")
    logging.info("Loading best model for final testing...")

    best_checkpoint_path = os.path.join(base_dir, "checkpoint_best_loss.pth.tar")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

        if test_dataloader is not None:
            logging.info("Final evaluation on test set.")
            eval_epoch("Final", test_dataloader, net, criterion)
        else:
            logging.info("Final evaluation on validation set.")
            eval_epoch("Final", val_dataloader, net, criterion)
    else:
        logging.warning("Could not find best checkpoint for final testing.")

    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])