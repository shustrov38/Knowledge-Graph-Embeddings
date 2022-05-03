import torch

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False);


class SME_bil(torch.jit.ScriptModule):
    
    def __init__(self, m: int, d: int, p: int):
        super().__init__()
        self.embed = torch.nn.Embedding(m, d, max_norm=1.0);

        self.Wl = torch.nn.Parameter(torch.rand((p, d, d)))
        self.Wr = torch.nn.Parameter(torch.rand((p, d, d)))

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
        
        expr = 'ijk,...k,...j->i'
        
        return - (torch.einsum(expr, self.Wl, rel.t(), lhs.t()) + self.bl.t())\
                        .matmul(
                            (torch.einsum(expr, self.Wr, rel.t(), rhs.t()) + self.br.t()).t()
                        )
        