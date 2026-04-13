"""Dual-LR optimizer for SA-initialized DFlash training.

fc (randomly initialized): high LR (default 6e-3) with cosine decay
layers (SA initialized): low LR (default 6e-4) with cosine decay
Both train from step 0, no freezing.
"""

import torch

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0


class PhasedBF16Optimizer:
    def __init__(
        self,
        draft_model,
        fc_lr: float = 6e-3,
        layer_lr: float = 6e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        total_steps: int = 100000,
        warmup_ratio: float = 0.04,
        # legacy compat
        phase1_ratio: float = 0.05,
        layer_lr_ratio: float = None,
    ):
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        self.current_step = 0

        # Separate fc params vs layer params
        self.fc_params = []
        self.layer_params = []

        for name, p in draft_model.named_parameters():
            if not p.requires_grad:
                continue
            if "fc." in name or "hidden_norm." in name:
                self.fc_params.append(p)
            else:
                self.layer_params.append(p)

        warmup_steps = max(int(warmup_ratio * total_steps), 1)
        print_on_rank0(f"DualLR Optimizer: {len(self.fc_params)} fc params (lr={fc_lr}), "
                       f"{len(self.layer_params)} layer params (lr={layer_lr})")
        print_on_rank0(f"  Both train from step 0, no freezing. Warmup={warmup_steps} steps.")

        # FP32 copies
        self.fc_fp32 = [p.detach().clone().to(torch.float32) for p in self.fc_params]
        self.layer_fp32 = [p.detach().clone().to(torch.float32) for p in self.layer_params]
        for p in self.fc_fp32 + self.layer_fp32:
            p.requires_grad = True

        # fc optimizer + scheduler
        self.fc_optimizer = torch.optim.AdamW(
            self.fc_fp32, lr=fc_lr, weight_decay=weight_decay
        )
        self.fc_scheduler = CosineAnnealingWarmupLR(
            self.fc_optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        # layer optimizer + scheduler
        self.layer_optimizer = torch.optim.AdamW(
            self.layer_fp32, lr=layer_lr, weight_decay=weight_decay
        )
        self.layer_scheduler = CosineAnnealingWarmupLR(
            self.layer_optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

    def step(self):
        self.current_step += 1

        # fc update
        with torch.no_grad():
            for p, fp in zip(self.fc_params, self.fc_fp32):
                fp.grad = p.grad.detach().to(torch.float32) if p.grad is not None else None
        torch.nn.utils.clip_grad_norm_(self.fc_fp32, self.max_grad_norm)
        self.fc_optimizer.step()
        self.fc_optimizer.zero_grad()
        self.fc_scheduler.step()

        # layer update
        with torch.no_grad():
            for p, fp in zip(self.layer_params, self.layer_fp32):
                fp.grad = p.grad.detach().to(torch.float32) if p.grad is not None else None
        torch.nn.utils.clip_grad_norm_(self.layer_fp32, self.max_grad_norm)
        self.layer_optimizer.step()
        self.layer_optimizer.zero_grad()
        self.layer_scheduler.step()

        # Copy back to bf16
        with torch.no_grad():
            for p, fp in zip(self.fc_params, self.fc_fp32):
                p.data.copy_(fp.data.to(p.dtype))
                p.grad = None
            for p, fp in zip(self.layer_params, self.layer_fp32):
                p.data.copy_(fp.data.to(p.dtype))
                p.grad = None

    def get_learning_rate(self):
        return self.fc_optimizer.param_groups[0]["lr"]

    def get_layer_learning_rate(self):
        return self.layer_optimizer.param_groups[0]["lr"]

    def get_phase(self):
        return 1

    def state_dict(self):
        return {
            "optimizer_state_dict": self.fc_optimizer.state_dict(),
            "layer_optimizer_state_dict": self.layer_optimizer.state_dict(),
            "scheduler_state_dict": self.fc_scheduler.state_dict(),
            "layer_scheduler_state_dict": self.layer_scheduler.state_dict(),
            "current_step": self.current_step,
            "phase": 1,
        }

    def load_state_dict(self, state_dict):
        self.fc_optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.layer_optimizer.load_state_dict(state_dict["layer_optimizer_state_dict"])
        self.fc_scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.layer_scheduler.load_state_dict(state_dict["layer_scheduler_state_dict"])
        self.current_step = state_dict["current_step"]
        print_on_rank0(f"Restored DualLR Optimizer: step={self.current_step}")
