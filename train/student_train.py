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
# 假設你的專案結構中有這個 module
from compressai.models.cic import build_model 
import wandb

# ==========================================================
# 知識蒸餾損失函數 (Knowledge Distillation Loss)
# L = alpha * L_feat + beta * L_recon + gamma * L_RD
# ==========================================================
class DistillationLoss(nn.Module):
    def __init__(self, lmbda=1e-2, roi_factor=0.1, alpha=0.3, beta=0.3, gamma=0.4):
        """
        Args:
            lmbda (float): 學生模型 RD Loss 中的 Rate-Distortion 平衡參數
            roi_factor (float): ROI (水域) 權重因子
            alpha (float): 特徵層級蒸餾權重 (Feature Loss)
            beta (float): 重建層級蒸餾權重 (Reconstruction Loss)
            gamma (float): 學生模型 RD 損失權重 (Student RD Loss)
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_pixel = nn.MSELoss(reduction='none') # Pixel-wise for ROI
        self.lmbda = lmbda
        self.roi_factor = roi_factor
        
        # Distillation weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def generate_water_mask(self, images):
        """ 延用 Teacher Train 中的 ROI 遮罩生成邏輯 """
        images = torch.clamp(images, 0, 1)
        r = images[:, 0, :, :]
        g = images[:, 1, :, :]
        b = images[:, 2, :, :]
        
        # RGB conditions
        cond1 = b > (r * 1.1)
        cond2 = b > g
        cond3 = b < 0.9
        
        # Saturation condition
        max_rgb = torch.max(torch.max(r, g), b)
        min_rgb = torch.min(torch.min(r, g), b)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        cond4 = saturation < 0.3
        
        is_water = cond1 & cond2 & cond3 & cond4
        weight_map = torch.ones_like(r)
        weight_map[is_water] = self.roi_factor
        
        return weight_map.unsqueeze(1)

    def forward(self, student_out, teacher_out, target):
        out = {}
        N, _, H, W = target.size()
        num_pixels = N * H * W

        # -----------------------------------------------------------
        # 1. Feature-Level Distillation (Latent Space y) 
        # -----------------------------------------------------------
        if "y_hat" not in teacher_out:
             raise KeyError("Teacher model output must contain 'y_hat' for feature distillation.")
        
        loss_distill_feature = self.mse(student_out["y_hat"], teacher_out["y_hat"])

        # -----------------------------------------------------------
        # 2. Reconstruction-Level Distillation (Output Image x_hat) 
        # -----------------------------------------------------------
        loss_distill_recon = self.mse(student_out["x_hat"], teacher_out["x_hat"])

        # -----------------------------------------------------------
        # 3. Student Rate-Distortion Loss (Standard LIC Loss)
        # -----------------------------------------------------------
        # Bit rate loss
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in student_out["likelihoods"].values()
        )

        # Weighted MSE (ROI)
        roi_weights = self.generate_water_mask(target)
        mse_pixel_wise = self.mse_pixel(student_out["x_hat"], target)
        weighted_mse = (mse_pixel_wise * roi_weights).mean()
        
        # RD Loss = R + lambda * D
        rd_loss = self.lmbda * 255**2 * weighted_mse + bpp_loss

        # -----------------------------------------------------------
        # Total Loss Calculation
        # L = alpha * L_feat + beta * L_recon + gamma * L_RD
        # -----------------------------------------------------------
        total_loss = (self.alpha * loss_distill_feature + 
                      self.beta * loss_distill_recon + 
                      self.gamma * rd_loss)

        out.update({
            "loss": total_loss,
            "distill_feature_loss": loss_distill_feature,
            "distill_recon_loss": loss_distill_recon,
            "student_rd_loss": rd_loss,
            "bpp_loss": bpp_loss,
            "mse_loss": weighted_mse,
            "mse_loss_raw": self.mse(student_out["x_hat"], target) # For PSNR
        })

        return out

class AverageMeter:
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
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def init(args):
    # 修改存檔路徑命名規則，加入 Student N 的資訊，方便區分 N64 或 N32
    base_dir = f'./pretrained/student/{args.student_model}_N{args.student_n}/q{args.quality_level}/L{args.lmbda:.4f}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除既有的 handlers 避免重複 log
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)
    
    logging.info('Logging file is %s' % log_dir)

def configure_optimizers(net, args):
    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    params_dict = dict(net.named_parameters())
    
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=args.learning_rate)
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer

def train_one_epoch(student, teacher, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm):
    student.train()
    teacher.eval() # Teacher 必須絕對凍結
    device = next(student.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # 1. Get Teacher Output (No Grad)
        with torch.no_grad():
            teacher_out = teacher(d)

        # 2. Get Student Output
        student_out = student(d)

        # 3. Compute Distillation Loss
        out_criterion = criterion(student_out, teacher_out, d)
        
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = student.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            mse_raw = out_criterion["mse_loss_raw"].item()
            psnr_val = 10 * math.log10(1.0 / (mse_raw + 1e-10))
            
            logging.info(
                f"Epoch [{epoch}][{i}/{len(train_dataloader)}] "
                f"Loss: {out_criterion['loss'].item():.4f} | "
                f"Feat: {out_criterion['distill_feature_loss'].item():.5f} | "
                f"Recon: {out_criterion['distill_recon_loss'].item():.5f} | "
                f"RD: {out_criterion['student_rd_loss'].item():.4f} | "
                f"PSNR: {psnr_val:.2f}"
            )
            
            wandb.log({
                "train_loss": out_criterion['loss'].item(),
                "distill_feature": out_criterion['distill_feature_loss'].item(),
                "distill_recon": out_criterion['distill_recon_loss'].item(),
                "student_rd": out_criterion['student_rd_loss'].item(),
                "train_psnr": psnr_val
            })

def eval_epoch(epoch, dataloader, model, criterion):
    # 評估時只看 Student 自己的表現 (PSNR/Bpp)
    model.eval()
    device = next(model.parameters()).device
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bpp_meter = AverageMeter()
    
    with torch.no_grad():
        for d in dataloader:
            d = d.to(device)
            out_net = model(d)
            
            # 這裡簡單計算 Student 自己的 RD 表現
            num_pixels = d.size(0) * d.size(2) * d.size(3)
            bpp = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in out_net["likelihoods"].values())
            
            mse_raw = F.mse_loss(out_net["x_hat"], d) # 使用 Raw MSE 計算 PSNR
            psnr = 10 * math.log10(1.0 / (mse_raw.item() + 1e-10))
            
            # 這裡我們用 MSE 作為驗證指標 (Loss 越小越好)
            loss_meter.update(mse_raw.item()) 
            psnr_meter.update(psnr)
            bpp_meter.update(bpp.item())

    logging.info(
        f"Eval Epoch {epoch}: PSNR: {psnr_meter.avg:.2f} | Bpp: {bpp_meter.avg:.4f}"
    )
    wandb.log({"val_psnr": psnr_meter.avg, "val_bpp": bpp_meter.avg})
    return loss_meter.avg

def save_checkpoint(state, is_best, base_dir):
    filename = "checkpoint.pth.tar"
    save_path = os.path.join(base_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(base_dir, "checkpoint_best.pth.tar"))

def parse_args(argv):
    parser = argparse.ArgumentParser(description="CIC Student Distillation Training")
    
    # Teacher/Student 架構控制參數
    parser.add_argument("--teacher-model", type=str, default="cic", help="Teacher model architecture name")
    parser.add_argument("--teacher-n", type=int, default=128, help="Teacher hidden channels N")
    
    parser.add_argument("--student-model", type=str, default="cic_student", help="Student model architecture name")
    parser.add_argument("--student-n", type=int, default=64, help="Student hidden channels N")
    
    parser.add_argument("--m", type=int, default=192, help="Latent channels M (Shared)")
    
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path")
    parser.add_argument("-t", "--teacher-checkpoint", type=str, required=True, help="Path to pre-trained teacher checkpoint")
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("-n", "--num-workers", type=int, default=32)
    parser.add_argument("-q", "--quality-level", type=int, default=3)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=1e-2, help="RD Trade-off parameter")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str)
    
    # Resume 參數
    parser.add_argument("--checkpoint", type=str, help="Path to student checkpoint (resume)")
    
    parser.add_argument("--roi-factor", type=float, default=0.1)
    
    # KD Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.3, help="Weight for Feature Loss")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for Recon Loss")
    parser.add_argument("--gamma", type=float, default=0.4, help="Weight for RD Loss")

    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    wandb.init(project="cic-student-distillation", name=args.name, config=vars(args))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    logging.info(f'Starting Student Training: {args.name}')

    # transforms
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    val_dataset = ImageFolder(args.dataset, split="val", transform=test_transforms)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(device=="cuda"))
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device=="cuda"))

    # =========================================================
    # 初始化 Teacher
    # =========================================================
    logging.info(f"Building Teacher Model: {args.teacher_model} (N={args.teacher_n}, M={args.m})")
    
    teacher = build_model(args.teacher_model, N=args.teacher_n, M=args.m)
    
    logging.info(f"Loading Teacher Checkpoint: {args.teacher_checkpoint}")
    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location="cpu")
    
    if "state_dict" in teacher_ckpt:
        teacher.load_state_dict(teacher_ckpt["state_dict"])
    else:
        teacher.load_state_dict(teacher_ckpt)
        
    teacher = teacher.to(device)
    teacher.eval() 

    # =========================================================
    # 初始化 Student
    # =========================================================
    logging.info(f"Building Student Model: {args.student_model} (N={args.student_n}, M={args.m})")
    
    student = build_model(args.student_model, N=args.student_n, M=args.m)
    student = student.to(device)
    logging.info(f"Student Initialized.")

    if args.cuda and torch.cuda.device_count() > 1:
        student = CustomDataParallel(student)

    # Optimizer
    optimizer, aux_optimizer = configure_optimizers(student, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 350], gamma=0.1)

    # Distillation Loss
    criterion = DistillationLoss(
        lmbda=args.lmbda, 
        roi_factor=args.roi_factor,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )

    # =========================================================
    # RESUME LOGIC (優化版)
    # =========================================================
    last_epoch = 0
    best_loss = float("inf") # 預設為無限大

    if args.checkpoint:
        logging.info(f"Resuming Student from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        last_epoch = checkpoint["epoch"] + 1
        student.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        # [NEW] 嘗試讀取之前的 best_loss，避免覆蓋最佳模型
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
            logging.info(f"Resuming with best_loss: {best_loss}")
        elif "loss" in checkpoint:
            # 如果沒有 best_loss，暫時用該 checkpoint 紀錄的 loss 當作基準
            best_loss = checkpoint["loss"]
            logging.warning(f"Warning: 'best_loss' not found. Using current checkpoint loss ({best_loss}) as baseline.")
        else:
            logging.warning("Warning: Neither 'best_loss' nor 'loss' found. Starting with inf.")

    # =========================================================
    # Training Loop
    # =========================================================
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            student, teacher, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm
        )
        
        val_loss = eval_epoch(epoch, val_dataloader, student, criterion)
        lr_scheduler.step()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": student.state_dict(),
                    "loss": val_loss,
                    "best_loss": best_loss, # [NEW] 保存 best_loss
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "N": args.student_n,
                    "M": args.m,
                },
                is_best,
                base_dir
            )

    wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])