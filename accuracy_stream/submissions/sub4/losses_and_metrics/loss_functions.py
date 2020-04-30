import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = torch.mean(
            self.mse(yhat, y).mean(1))
        return loss


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = torch.mean(
            torch.sqrt(self.mse(yhat, y).mean(1)))
        return loss


class RMSSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = torch.mean(
            torch.sqrt(self.mse(yhat, y).mean(1) / scale))
        return loss


class WRMSSELevel12Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, yhat, y, scale, weight):
        loss = torch.sum(
            weight * torch.sqrt(self.mse(yhat, y).mean(1) / scale))
        return loss

