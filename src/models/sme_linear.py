import torch

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False);


class SME_lin(torch.jit.ScriptModule):
    
    def __init__(self, m: int, d: int, p: int):
        super().__init__()
        self.embed = torch.nn.Embedding(m, d, max_norm=1.0)
        
        self.Wl1 = torch.nn.Parameter(torch.rand((p, d)))
        self.Wl2 = torch.nn.Parameter(torch.rand((p, d)))
        self.Wr1 = torch.nn.Parameter(torch.rand((p, d)))
        self.Wr2 = torch.nn.Parameter(torch.rand((p, d)))
                                      
        self.bl = torch.nn.Parameter(torch.zeros(p))
        self.br = torch.nn.Parameter(torch.zeros(p))
    
    @torch.jit.script_method
    def forward(self, normal: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:       
        # pairwise hinge loss with margin = 1
        return torch.max(self.energy(normal) - self.energy(negative) + torch.ones(1), torch.zeros(1))
    
    @torch.jit.script_method
    def energy(self, triplet: torch.Tensor) -> torch.Tensor:
        lhs = self.embed(triplet[0])
        rel = self.embed(triplet[1])
        rhs = self.embed(triplet[2])
        
        return - (self.Wl1.matmul(lhs.t()) + self.Wl2.matmul(rel.t()) + self.bl.t()) \
                        .matmul(
                            (self.Wr1.matmul(rhs.t()) + self.Wr2.matmul(rel.t()) + self.br.t()).t()
                        )
    