import torch

# Pytorch loss functions
LOSS_FUNCTIONS = {
    "l1loss": torch.nn.L1Loss,
    "mseloss": torch.nn.MSELoss,
    "crossentropyloss": torch.nn.CrossEntropyLoss,
    "ctcloss": torch.nn.CTCLoss,
    "nllloss": torch.nn.NLLLoss,
    "poissonnllloss": torch.nn.PoissonNLLLoss,
    "gaussiannllloss": torch.nn.GaussianNLLLoss,
    "kldivloss": torch.nn.KLDivLoss,
    "bceloss": torch.nn.BCELoss,
    "bcewithlogitsloss": torch.nn.BCEWithLogitsLoss,
    "marginrankingloss": torch.nn.MarginRankingLoss,
    "hingeembeddingloss": torch.nn.HingeEmbeddingLoss,
    "multilabelmarginloss": torch.nn.MultiLabelMarginLoss,
    "huberloss": torch.nn.HuberLoss,
    "smoothl1loss": torch.nn.SmoothL1Loss,
    "softmarginloss": torch.nn.SoftMarginLoss,
    "multilabelsoftmarginloss": torch.nn.MultiLabelSoftMarginLoss,
    "cosineembeddingloss": torch.nn.CosineEmbeddingLoss,
    "multimarginloss": torch.nn.MultiMarginLoss,
    "tripletmarginloss": torch.nn.TripletMarginLoss,
    "tripletmarginwithdistanceloss": torch.nn.TripletMarginWithDistanceLoss,
}

class RMSELoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mse = torch.nn.MSELoss(*args, **kwargs)
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

LOSS_FUNCTIONS["rmseloss"] = RMSELoss



class plusDerivative(torch.nn.Module):
    # wrapper that adds loss of derivatives to regular loss. Both loss functionals used can be specified independently.
    def __init__(self, *args, functional = None, derivative_functional = None, **kwargs):
        if derivative_functional == None:
            derivative_functional = functional
        if functional == None:
            raise ValueError("Missing functional for _loss_functions.plusDerivative() metric. Please include \"functional\" kwarg.")
        super().__init__()
        self.functional = LOSS_FUNCTIONS[functional.lower()](*args, **kwargs)
        self.der_functional = LOSS_FUNCTIONS[derivative_functional.lower()](*args, **kwargs)

    def forward(self, yhat, y):
        return self.functional(yhat, y)+self.der_functional(yhat[1:]-yhat[:-1], y[1:]-y[:-1])

LOSS_FUNCTIONS["plusderivative"] = plusDerivative