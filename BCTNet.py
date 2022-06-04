class BC(nn.Module):
    # Bole convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.Sigmoid()
        self.act1 = nn.Tanh()
    def forward(self, x):
        reward = 2.56 * self.act(self.conv(x))
        punishment = 0.39 * self.act(self.conv(x))
        rp = torch.where(self.act(self.conv(x)) > 0.4, reward, punishment)
        ni = self.act1(self.conv(x))
        bc = torch.multiply(rp, ni)
        return bc

class CAM(nn.Module):

    def __init__(self, in_dim, *args):  #,**kwargs
        super().__init__()
        self.dim = in_dim
        self.Horizontal_Convd = nn.Conv2d(in_channels=in_dim,out_channels=2*in_dim,kernel_size=3,stride=1,padding=1)
        self.vertical_Convd = nn.Conv2d(in_channels=in_dim,out_channels=2*in_dim,kernel_size=3,stride=1,padding=1)
        self.Horizontal_Convu = nn.Conv2d(in_channels=2*in_dim,out_channels=in_dim,kernel_size=3,stride=1,padding=1)
        self.vertical_Convu = nn.Conv2d(in_channels=2*in_dim,out_channels=in_dim,kernel_size=3,stride=1,padding=1)
        self.Sakura_Mint = nn.Parameter(torch.zeros(1))
        self.x_dim = nn.Softmax(dim=1)
        self.H_dim = nn.Softmax(dim=2)
        self.V_dim = nn.Softmax(dim=3)

    def forward(self, x):
        # x is a list (Feature matrix, Laplacian (Adjcacency) Matrix).
        #assert isinstance(x, list)
        #x = torch.tensor(x)
        _,C_dim1,H_dim2,W_dim3 = x.shape
        #print(x.size())
        W_Vertical = x.permute(0,1,3,2).contiguous()
        W_Vertical = self.vertical_Convd(W_Vertical)
        W_Vertical = torch.relu(W_Vertical)
        W_Vertical = self.vertical_Convu(W_Vertical)
        Return_W = self.V_dim(W_Vertical)
        #a = a.reshape(W_dim3,H_dim2,C_dim1)
        H_horizontal = x.permute(0,1,2,3).contiguous()
        H_horizontal = self.Horizontal_Convd(H_horizontal)
        H_horizontal = torch.relu(H_horizontal)
        H_horizontal = self.Horizontal_Convu(H_horizontal)
        Return_H = self.H_dim(H_horizontal)
        return self.Sakura_Mint*(Return_H+Return_W)

class Convup(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Convup, self).__init__()
        self.convTrans = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1= nn.Conv2d(32, 256, kernel_size=1)
    def forward(self, x):
            x[1] = self.conv1(x[1])
            x = torch.stack(x, 0)
            k = self.act(self.bn(self.convTrans(x)))
            return k

class BTFPNcat1(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.up = Convup(c1=1, c2=1)
    def forward(self, x):
        k = self.up(x)
        return torch.cat(k, self.d)


class Convdo(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Convdo, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        l = self.act(self.bn(self.convTrans(x)))
        return l

class BTFPNcat2(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.up = Convdo(c1=1, c2=1)
    def forward(self, x):
        l = self.up(x)
        return torch.cat(l, self.d)
