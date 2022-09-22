
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable

from genotypes import Genotype

PRIMITIVES = [
    'none',
    'avg_pool_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

class MixedOp(nn.Module):
    def __init__(self, C, stride) -> None:
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C,stride,False)
            if ('pool' in primitive):
                op = nn.Sequential(op, nn.BatchNorm2d(C,affine = False))
            self._ops.append(op)

    def forward(self, x , weights):
        # print (x.size())
        # print (len((self._ops)))
        # print (weights.size())

        # for w,op in zip(weights, self._ops):
        #     print (op(x).size())

        return sum(w * op(x) for w,op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, steps, multplier, C_prev_prev, C_prev, C, reduction, reduction_prev) -> None:
        super(Cell, self).__init__()

        self.reduction = reduction
        if (reduction_prev):
            self.preprocess0 = FactorizedReduce(C_prev_prev,C, affine = False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine = False)
        
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine = False)
        self._steps = steps
        self._multiplier = multplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j<2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        # print ("Cell Debugging!")
        # print (weights.size())
        s0 = self.preprocess0(s0) # C_pp -> C
        s1 = self.preprocess1(s1) # C_p -> C
        # print (s0.size())
        # print (s1.size())
        states = [s0,s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j,h in enumerate(states))
            # print (s)
            # exit()
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim = 1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps = 4, multiplier = 4, stem_multiplier = 3):
        super(Network, self).__init__()

        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, kernel_size = 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(C_curr)
        )

        self.cells = nn.ModuleList()

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3,2*layers//3]: # if current layer is in between //3 & 2//3
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev = C_prev
            C_prev = multiplier * C_curr

        # Define Head
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self.initialize_alphas()

    def initialize_alphas(self):
        # ------- Initialize alphas --------
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad = True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad = True)

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce
        ]


    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x,y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def forward(self, input):
        # STem
        s0 = s1 = self.stem(input)

        # print ("S0 : " + str(s0.size()))
        # print ("S1 : " + str(s1.size()))

        # Intermediate
        for i, cell in enumerate(self.cells):
            if (cell.reduction):
                weights = F.softmax(self.alphas_reduce, dim = -1)
            else:
                weights = F.softmax(self.alphas_normal, dim = -1)
            s0,s1 = s1, cell(s0,s1,weights)
            # print ("Cell S0 : " + str(s0.size()))
            # print ("Cell S1 : " + str(s1.size()))

        # exit()
        # Head
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key = lambda x: -max( W[x][k] for k in range(len(W[x])) if k!= PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if (k!=PRIMITIVES.index('none')):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        
        gene_normal = _parse(F.softmax(self.alphas_normal, dim = -1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim = -1).data.cpu().numpy())
        concat = range(2 + self._steps - self._multiplier, 2 + self._steps)
        genotype = Genotype(
            normal = gene_normal,
            normal_concat= concat,
            reduce = gene_reduce,
            reduce_concat = concat
        )

        return genotype