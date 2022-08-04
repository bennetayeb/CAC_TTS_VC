import torch

class cca_loss():
    def __init__(self, out_dim, device, use_all_singvals=False):
        self.out_dim = out_dim  # parameter o in original paper
        self.use_all_singvals = use_all_singvals
        self.device = device

    def loss(self, H1, H2):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        # transpose H1, H2: m x o -> o x m
        H1 = H1.t()
        H2 = H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        m = H1.size(1)
        o1, o2 = H1.size(0), H2.size(0)

        # produce the centered data matrices: H1 - 1/m*H1·I (same for H2bar)
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        assert torch.isnan(H1bar).sum().item() == 0
        assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # calculate the root inverse (e.g. SigmaHat11^(-1/2)) using sigular value decomposition
        D1, V1 = torch.symeig(SigmaHat11, eigenvectors=True)
        D2, V2 = torch.symeig(SigmaHat22, eigenvectors=True)

        # ??? probably problemetic in gene count setting
        posIdx1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posIdx1]
        V1 = V1[:, posIdx1]

        posIdx2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posIdx2]
        V2 = V2[:, posIdx2]

        # ???take care of torch.sqrt
        SigmaHatRootInv11 = torch.matmul(torch.matmul(V1, torch.diag((D1) ** (-0.5))), V1.t())
        SigmaHatRootInv22 = torch.matmul(torch.matmul(V2, torch.diag((D2) ** (-0.5))), V2.t())

        # calculate T
        Tval = torch.matmul(torch.matmul(SigmaHatRootInv11, SigmaHat12), SigmaHatRootInv22)

        # calculate corr(H1, H2): matrix trace norm of T or sum of top k singular vals of T
        trace_TT = torch.matmul(Tval.t(), Tval)
        if self.use_all_singvals:
            corr = torch.trace(torch.sqrt(trace_TT))
            # assert torch.isnan(corr).item() == 0
        else:
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0]) * r1).to(self.device))
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U > eps, U, (torch.ones(U.shape).float() * eps).to(self.device))
            U = U.topk(self.out_dim)[0]
            corr = torch.sum(torch.sqrt(U))
            # print("loss: " + str(-corr))
        return -corr