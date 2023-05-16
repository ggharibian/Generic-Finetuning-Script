import torch

class Custom_Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Insert layers here.
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Insert forward pass on layers here.
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(x)
        # TODO: Insert prediction logic here.
        return prediction