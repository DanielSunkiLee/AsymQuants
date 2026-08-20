import torch

class AsymmetricFakeQuants(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,group_size=32):
        original_shape = x.shape
        x_grouped = x.contiguous().view(-1,group_size)
        # torch.min/max return a namedtuple (values,indices)
        # no need indices' location. 
        min_val,_ = torch.min(x_grouped, dim=-1, keepdim=True)
        max_val,_ = torch.max(x_grouped, dim=-1, keepdim=True)

        scale = (max_val-min_val)/255.0
        scale = torch.clamp(scale,min=1e-8)

        z = torch.round(-min_val/scale)-128.0

        scaled = torch.round(x_grouped/scale) + z
        clipped = torch.clamp(scaled,-128,127)

        reconstructed = (clipped-z)*scale
        return reconstructed.view(original_shape)
    
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
batch_size = 2
seq_len = 128
hidden_dim = 512

x = torch.randn(
    (batch_size,seq_len,hidden_dim),
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

weight = torch.randn(
    (hidden_dim,hidden_dim),
    dtype=torch.float32,
    device=device,
    requires_grad=True  
)


#quantized_weights = AsymmetricFakeQuants.apply(weight)

#Matrix Multiplication x @ W^T
y = torch.matmul(x, weight.t())

#원래 FP32 x 와 reconstructed  사이의 오차 계산 코드 작성
target = torch.randn_like(y)
loss = torch.mean((y-target)**2)

loss.backward()

print(f"without quantized {weight.grad}")