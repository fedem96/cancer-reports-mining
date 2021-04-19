from torch.autograd import Function


class GradientReversal(Function):

    @staticmethod
    def forward(ctx, x, l):
        ctx.l = l
        return x

    @staticmethod
    def backward(ctx, grad):
        return -ctx.l * grad, None


gradient_reversal = GradientReversal.apply
