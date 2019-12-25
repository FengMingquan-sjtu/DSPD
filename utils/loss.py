import torch
import torch.nn as nn
import torch.nn.functional as F

class L1_BCE(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_BCE, self).__init__()
        weights={"corners":200,"edges":16}
        self.corners_loss=nn.BCEWithLogitsLoss(weight=torch.tensor(weights["corners"]))
        self.edges_loss=nn.BCEWithLogitsLoss(weight=torch.tensor(weights["edges"]))
        self.SR_loss=nn.L1Loss(reduction="sum")

        

    def forward(self, X, Y):
        corners_loss=self.corners_loss(X[:,0],Y[:,0])
        edges_loss=self.edges_loss(X[:,1],Y[:,1])
        SR_loss=self.SR_loss(X[:,2:],Y[:,2:])
        return torch.add(torch.add(corners_loss,edges_loss),SR_loss)

