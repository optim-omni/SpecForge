"""Two-phase optimizer for SA-initialized DFlash training.

Phase 1 (0-5% steps):   Train fc only, lr=6e-3. Quick alignment.
Phase 2 (5-100% steps):  Train fc + layers. Layers lr=6e-4, fc follows layers lr.
Both phases use cosine decay with warmup.
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
        phase1_ratio: float = 0.05,
        # legacy compat
        layer_lr_ratio: float = None,
    ):
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        self.current_step = 0

        # Phase boundary
        self.phase1_end = int(total_steps * phase1_ratio)

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

        print_on_rank0(f"PhasedOptimizer: {len(self.fc_params)} fc params, {len(self.layer_params)} layer params")
        print_on_rank0(f"  Phase 1 (step 0-{self.phase1_end}): fc only, fc_lr={fc_lr}")
        print_on_rank0(f"  Phase 2 (step {self.phase1_end}-{total_steps}): fc + layers, layer_lr={layer_lr}, fc follows layer_lr")

        # FP32 copies
        self.fc_fp32 = [p.detach().clone().to(torch.float32) for p in self.fc_params]
        self.layer_fp32 = [p.detach().clone().to(torch.float32) for p in self.layer_params]
        for p in self.fc_fp32 + self.layer_fp32:
            p.requires_grad = True

        # Phase 1: fc only, high LR with cosine decay to layer_lr
        self.fc_optimizer = torch.optim.AdamW(
            self.fc_fp32, lr=fc_lr, weight_decay=weight_decay
        )
        self.phase1_scheduler = CosineAnnealingWarmupLR(
            self.fc_optimizer,
            total_steps=self.phase1_end,
            warmup_steps=max(int(warmup_ratio * self.phase1_end), 1),
            eta_min=layer_lr,
        )

        # Phase 2: layers with own LR, fc follows
        self.layer_optimizer = torch.optim.AdamW(
            self.layer_fp32, lr=layer_lr, weight_decay=weight_decay
        )
        phase2_steps = total_steps - self.phase1_end
        self.phase2_scheduler = CosineAnnealingWarmupLR(
            self.layer_optimizer,
            total_steps=phase2_steps,
            warmup_steps=max(int(warmup_ratio * phase2_steps), 1),
        )

        # Initially freeze layers
        for p in self.layer_params:
            p.requires_grad = False

        self._phase = 1

    def _transition_to_phase2(self):
        print_on_rank0(f"  >>> Phase 2 at step {self.current_step}: unfreeze layers")
        for p in self.layer_params:
            p.requires_grad = True
        # Set fc LR to layer LR
        layer_lr = self.layer_optimizer.param_groups[0]["lr"]
        for pg in self.fc_optimizer.param_groups:
            pg["lr"] = layer_lr
        self._phase = 2

    def step(self):
        self.current_step += 1

        if self._phase == 1 and self.current_step >= self.phase1_end:
            self._transition_to_phase2()

        # --- fc update (all phases) ---
        with torch.no_grad():
            for p, fp in zip(self.fc_params, self.fc_fp32):
                fp.grad = p.grad.detach().to(torch.float32) if p.grad is not None else None
        torch.nn.utils.clip_grad_norm_(self.fc_fp32, self.max_grad_norm)
        self.fc_optimizer.step()
        self.fc_optimizer.zero_grad()

        if self._phase == 1:
            self.phase1_scheduler.step()
        else:
            # Phase 2: fc LR follows layer LR
            layer_lr = self.layer_optimizer.param_groups[0]["lr"]
            for pg in self.fc_optimizer.param_groups:
                pg["lr"] = layer_lr

        # --- layer update (phase 2 only) ---
        if self._phase == 2:
            with torch.no_grad():
                for p, fp in zip(self.layer_params, self.layer_fp32):
                    fp.grad = p.grad.detach().to(torch.float32) if p.grad is not None else None
            torch.nn.utils.clip_grad_norm_(self.layer_fp32, self.max_grad_norm)
            self.layer_optimizer.step()
            self.layer_optimizer.zero_grad()
            self.phase2_scheduler.step()

        # --- Copy back to bf16 ---
        with torch.no_grad():
            for p, fp in zip(self.fc_params, self.fc_fp32):
                p.data.copy_(fp.data.to(p.dtype))
                p.grad = None
            if self._phase == 2:
                for p, fp in zip(self.layer_params, self.layer_fp32):
                    p.data.copy_(fp.data.to(p.dtype))
                    p.grad = None

    def get_learning_rate(self):
        return self.fc_optimizer.param_groups[0]["lr"]

    def get_layer_learning_rate(self):
        return self.layer_optimizer.param_groups[0]["lr"] if self._phase == 2 else 0.0

    def get_phase(self):
        return self._phase

    def state_dict(self):
        return {
            "optimizer_state_dict": self.fc_optimizer.state_dict(),
            "layer_optimizer_state_dict": self.layer_optimizer.state_dict(),
            "scheduler_state_dict": self.phase1_scheduler.state_dict() if self._phase == 1 else self.phase2_scheduler.state_dict(),
            "current_step": self.current_step,
            "phase": self._phase,
        }

    def load_state_dict(self, state_dict):
        self.fc_optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.layer_optimizer.load_state_dict(state_dict["layer_optimizer_state_dict"])
        self.current_step = state_dict["current_step"]
        self._phase = state_dict["phase"]
        if self._phase == 2:
            for p in self.layer_params:
                p.requires_grad = True
            self.phase2_scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0(f"Restored PhasedOptimizer: phase={self._phase}, step={self.current_step}")
