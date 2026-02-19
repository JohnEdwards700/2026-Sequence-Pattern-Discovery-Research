import torch
from torch.optim import Optimizer

class Scheduler:
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, initial_lr: float, final_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.initial_lr + (self.final_lr - self.initial_lr) * (self.current_step / self.warmup_steps)
        return self.final_lr * (1 - (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps))

    def reset(self):
        self.current_step = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr