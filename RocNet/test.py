from util import get_quad_feas
import torch

a = torch.randint(2,(512,512))
print(a.shape)
fea,op = get_quad_feas(a,32)
print(fea.shape)
print(op.shape)
