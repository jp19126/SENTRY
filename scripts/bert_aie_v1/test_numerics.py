"""Focused exact arithmetic checks independent of BERT quality acceptance."""
import unittest
import torch
from model import integer_dot, quantize, bf16_matmul

class NumericalContract(unittest.TestCase):
    def test_rounding_and_saturation(self):
        values=torch.tensor([-200.,-127.5,-126.5,-2.5,-1.5,-.5,.5,1.5,2.5,126.5,127.5,200.])
        self.assertEqual(quantize(values,1.).tolist(),[-127,-127,-126,-2,-2,0,0,2,2,126,127,127])
        self.assertEqual(quantize(torch.tensor([.25,.75,-.75]),.5).tolist(),[0,2,-2])

    def test_exact_dot_tails_and_accumulator(self):
        # Shapes intentionally exercise unaligned input/output counts and >int16 sums.
        gen=torch.Generator().manual_seed(17)
        x=torch.randint(-127,128,(2,3,67),generator=gen,dtype=torch.int8)
        w=torch.randint(-127,128,(5,67),generator=gen,dtype=torch.int8)
        expected=torch.tensor([[[sum(int(x[b,m,k])*int(w[o,k]) for k in range(67)) for o in range(5)] for m in range(3)] for b in range(2)],dtype=torch.int32)
        self.assertTrue(torch.equal(integer_dot(x,w),expected))
        full=torch.full((1,1024),127,dtype=torch.int8)
        self.assertEqual(int(integer_dot(full,full)[0,0]),16516096)
        if torch.cuda.is_available(): self.assertTrue(torch.equal(integer_dot(x.cuda(),w.cuda()).cpu(),expected))

    def test_bf16_operands_and_k_tail(self):
        # Exactly representable BF16 operands; independent scalar reduction.
        a=torch.tensor([[1.,2.,-3.,.5,4.]])
        b=torch.tensor([[2.,1.],[3.,-1.],[4.,2.],[8.,4.],[-2.,3.]])
        self.assertEqual(bf16_matmul(a,b,chunk=3).tolist(),[[-8.,7.]])

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
