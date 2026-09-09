"""Small policy/value networks, current-decision-owner features and hex symmetries."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gipf_engine as ge

ACTIONS = 42 + 21 * 128
PLANES = 9
GEO = ge.geometry()
COORDS = np.asarray(GEO['coordinates'], dtype=np.int64)
ROWS, COLS = COORDS[:, 1] + 3, COORDS[:, 0] + 3
MASK = np.zeros((7,7),np.float32); MASK[ROWS,COLS]=1

@dataclass
class ModelConfig:
    kind: str = 'resnet'
    width: int = 48
    blocks: int = 3
    head: str = 'flat'

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(4,c),nn.ReLU(),nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(4,c))
    def forward(self,x): return torch.relu(x+self.net(x))


class HexConv2d(nn.Conv2d):
    """Axial-grid 3x3 convolution with its two non-neighbour corners masked.

    Rows encode axial ``r`` and columns encode axial ``q``.  The top-left and
    bottom-right kernel offsets change both q and r in the same direction and
    are not adjacent hexes.  Multiplication in ``forward`` keeps their effect
    and gradients zero even if an optimizer stores nonzero weight values.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        if kernel_size != 3:
            raise ValueError('HexConv2d only represents 3x3 axial kernels')
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        mask = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        mask[0, 0, 0, 0] = mask[0, 0, 2, 2] = 0
        self.register_buffer('hex_mask', mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.hex_mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class HexResidual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            HexConv2d(c, c, 3, padding=1), nn.GroupNorm(4, c), nn.ReLU(),
            HexConv2d(c, c, 3, padding=1), nn.GroupNorm(4, c))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class TransformerBody(nn.Module):
    """Noncausal attention over the 37 playable cells and centered axial coords."""
    def __init__(self, width, blocks, heads=4, feedforward=128):
        super().__init__()
        if width % heads:
            raise ValueError(f'transformer width {width} must be divisible by {heads} heads')
        self.token = nn.Linear(PLANES, width)
        self.coordinate = nn.Linear(2, width, bias=False)
        layer = nn.TransformerEncoderLayer(width, heads, feedforward,
                                           dropout=0.0, activation='gelu',
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, blocks)
        coords = torch.tensor(COORDS, dtype=torch.float32).div_(3).unsqueeze(0)
        self.register_buffer('coordinates', coords)

    def forward(self, x):
        # All global feature planes are already present at every board token.
        tokens = x[:, :, ROWS, COLS].transpose(1, 2)
        tokens = self.token(tokens) + self.coordinate(self.coordinates)
        return self.encoder(tokens).flatten(1)

class PolicyValue(nn.Module):
    def __init__(self,config=ModelConfig()):
        super().__init__();self.config=config
        if config.kind=='mlp':
            w=config.width
            layers=[nn.Flatten(),nn.Linear(PLANES*49,w),nn.ReLU()]
            for _ in range(config.blocks): layers += [nn.Linear(w,w),nn.ReLU()]
            self.body=nn.Sequential(*layers);dim=w
        elif config.kind=='resnet':
            # Keep this literal sequence unchanged: existing ResNet checkpoint
            # keys and tensor shapes remain compatible.
            c=config.width
            self.body=nn.Sequential(nn.Conv2d(PLANES,c,3,padding=1),nn.GroupNorm(4,c),nn.ReLU(),*[Residual(c) for _ in range(config.blocks)],nn.Conv2d(c,8,1),nn.ReLU(),nn.Flatten())
            dim=8*49
        elif config.kind=='hexresnet':
            c=config.width
            self.body=nn.Sequential(HexConv2d(PLANES,c,3,padding=1),nn.GroupNorm(4,c),nn.ReLU(),*[HexResidual(c) for _ in range(config.blocks)],nn.Conv2d(c,8,1),nn.ReLU(),nn.Flatten())
            dim=8*49
        elif config.kind=='transformer':
            self.body=TransformerBody(config.width, config.blocks)
            dim=37*config.width
        else:
            raise ValueError(f'unknown model kind {config.kind!r}')
        self.policy=nn.Linear(dim,ACTIONS if config.head=='flat' else 100)
        if config.head=='factorized':
            # Captures factor into row choice + individual double-removal choices.
            # This shares learning across rare combinations of stacked pieces.
            coeff=np.zeros((21*128,37),np.float32)
            for lid,line in enumerate(GEO['lines']):
                for bits in range(128):
                    for j,cell in enumerate(line):
                        if bits&(1<<j):coeff[lid*128+bits,cell]=1
            self.register_buffer('capture_coeff',torch.tensor(coeff))
        self.value=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,1),nn.Tanh())
    def forward(self,x):
        h=self.body(x);raw=self.policy(h)
        if self.config.head=='factorized':
            captures=raw[:,42:63].repeat_interleave(128,dim=1)+raw[:,63:100]@self.capture_coeff.T
            raw=torch.cat((raw[:,:42],captures),dim=1)
        return raw,self.value(h).squeeze(-1)

def encode(state):
    b=np.asarray(state.board,dtype=np.int8)*state.current_player
    out=np.zeros((PLANES,7,7),np.float32)
    for i,p in enumerate((1,2,-1,-2)):out[i,ROWS,COLS]=(b==p)
    out[4]=MASK
    me=0 if state.current_player==1 else 1
    out[5]=MASK*(state.reserves[me]/18)
    out[6]=MASK*(state.reserves[1-me]/18)
    out[7]=MASK*(state.phase=='capture')
    out[8]=MASK*(state.turn_player==state.current_player)
    return out

# Exact dihedral action permutations preserve line-position mask bits.
def symmetries():
    coords=[tuple(x) for x in GEO['coordinates']];idx={v:i for i,v in enumerate(coords)}
    lines=GEO['lines'];rays=GEO['rays']
    # Contract rays may carry metadata; tolerate plain index arrays.
    rays=[r['cells'] if isinstance(r,dict) else r for r in rays]
    line_lookup={frozenset(l):i for i,l in enumerate(lines)}
    ray_lookup={tuple(r):i for i,r in enumerate(rays)}
    result=[]
    for reflect in (False,True):
        for rotations in range(6):
            perm=[]
            for q,r in coords:
                if reflect:q,r=r,q
                for _ in range(rotations):q,r=-r,q+r
                perm.append(idx[q,r])
            ap=np.arange(ACTIONS,dtype=np.int64)
            for a,ray in enumerate(rays):ap[a]=ray_lookup[tuple(perm[i] for i in ray)]
            for lid,line in enumerate(lines):
                newlid=line_lookup[frozenset(perm[i] for i in line)]
                target=lines[newlid];positions=[target.index(perm[i]) for i in line]
                for bits in range(128):
                    newbits=sum(1<<positions[j] for j in range(len(line)) if bits&(1<<j))
                    ap[42+128*lid+bits]=42+128*newlid+newbits
            result.append((np.asarray(perm),ap))
    return result

SYMMETRIES=symmetries()

def augment(x,actions,rng):
    perm,ap=SYMMETRIES[int(rng.integers(12))]
    out=np.zeros_like(x);out[:,ROWS[perm],COLS[perm]]=x[:,ROWS,COLS]
    return out,ap[actions]

def load_model(path,device='cpu'):
    data=torch.load(path,map_location=device,weights_only=False)
    m=PolicyValue(ModelConfig(**data['config'])).to(device)
    m.load_state_dict(data['model']);m.eval()
    return m,data
