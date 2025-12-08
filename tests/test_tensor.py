import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from somegrad.tensor import Tensor
import somegrad.functional as F

class TestTensor(unittest.TestCase):

    def test_init(self):
        data = np.array([1, 2, 3])
        t = Tensor(data)
        np.testing.assert_array_equal(t.data, data)
        self.assertEqual(t.device, 'cpu')
        self.assertEqual(t.shape, (3,))

        t2 = Tensor([1, 2, 3])
        np.testing.assert_array_equal(t2.data, data)

        with self.assertRaises(ValueError):
            Tensor(data, device='cuda')

    def test_properties(self):
        data = np.array([[1, 2], [3, 4]])
        t = Tensor(data)
        self.assertEqual(t.shape, (2, 2))
        
        new_data = np.array([[5, 6], [7, 8]])
        t.data = new_data
        np.testing.assert_array_equal(t.data, new_data)

    def test_add(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        t3 = t1 + t2
        np.testing.assert_array_equal(t3.data, [5, 7, 9])
        self.assertEqual(t3._op, '+')
        
        # Test broadcasting
        t4 = Tensor([1])
        t5 = t1 + t4
        np.testing.assert_array_equal(t5.data, [2, 3, 4])
        
        # Test add constant
        t6 = t1 + 1
        np.testing.assert_array_equal(t6.data, [2, 3, 4])
        
        # Test radd
        t7 = 1 + t1
        np.testing.assert_array_equal(t7.data, [2, 3, 4])

    def test_mul(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        t3 = t1 * t2
        np.testing.assert_array_equal(t3.data, [4, 10, 18])
        self.assertEqual(t3._op, '*')
        
        # Test constant
        t4 = t1 * 2
        np.testing.assert_array_equal(t4.data, [2, 4, 6])
        
        # Test rmul
        t5 = 2 * t1
        np.testing.assert_array_equal(t5.data, [2, 4, 6])

    def test_matmul(self):
        t1 = Tensor([[1, 2], [3, 4]])
        t2 = Tensor([[5, 6], [7, 8]])
        t3 = t1 @ t2
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(t3.data, expected)
        self.assertEqual(t3._op, '@')

    def test_pow(self):
        t1 = Tensor([1, 2, 3])
        t2 = t1 ** 2
        np.testing.assert_array_equal(t2.data, [1, 4, 9])
        self.assertEqual(t2._op, '**')
        
        t3 = t1 ** Tensor([2, 2, 2])
        np.testing.assert_array_equal(t3.data, [1, 4, 9])

    def test_neg(self):
        t1 = Tensor([1, -2, 3])
        t2 = -t1
        np.testing.assert_array_equal(t2.data, [-1, 2, -3])

    def test_sub(self):
        t1 = Tensor([4, 5, 6])
        t2 = Tensor([1, 2, 3])
        t3 = t1 - t2
        np.testing.assert_array_equal(t3.data, [3, 3, 3])
        
        t4 = t1 - 1
        np.testing.assert_array_equal(t4.data, [3, 4, 5])
        
        t5 = 10 - t1
        np.testing.assert_array_equal(t5.data, [6, 5, 4])

    def test_div(self):
        t1 = Tensor([4., 6., 8.])
        t2 = Tensor([2., 2., 2.])
        t3 = t1 / t2
        np.testing.assert_array_equal(t3.data, [2., 3., 4.])
        
        t4 = t1 / 2
        np.testing.assert_array_equal(t4.data, [2., 3., 4.])
        
        t5 = 12 / t1
        np.testing.assert_array_equal(t5.data, [3., 2., 1.5])

    def test_comparison(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 2, 2])
        
        self.assertTrue((t1 == t1).data.all())
        np.testing.assert_array_equal((t1 == t2).data, [False, True, False])
        np.testing.assert_array_equal((t1 != t2).data, [True, False, True])
        np.testing.assert_array_equal((t1 < t2).data, [True, False, False])
        np.testing.assert_array_equal((t1 > t2).data, [False, False, True])
        np.testing.assert_array_equal((t1 <= t2).data, [True, True, False])
        np.testing.assert_array_equal((t1 >= t2).data, [False, True, True])

    def test_functional_wrappers(self):
        t = Tensor([-1.0, 0.0, 1.0])
        
        # abs
        np.testing.assert_array_equal(abs(t).data, [1.0, 0.0, 1.0])
        
        # exp
        np.testing.assert_allclose(t.exp().data, np.exp(t.data))
        
        # log
        t_pos = Tensor([1.0, np.e])
        np.testing.assert_allclose(t_pos.log().data, [0.0, 1.0])
        
        # relu
        np.testing.assert_array_equal(t.relu().data, [0.0, 0.0, 1.0])
        
        # sum
        t2 = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t2.sum().data, 10)
        np.testing.assert_array_equal(t2.sum(axis=0).data, [4, 6])
        
        # mean
        self.assertEqual(t2.mean().data, 2.5)

        # reshape
        t3 = Tensor(np.arange(6))
        np.testing.assert_array_equal(t3.reshape(2, 3).shape, (2, 3))

        # tanh
        t_tanh = Tensor([0.0])
        np.testing.assert_array_equal(t_tanh.tanh().data, [0.0])

        # log10
        t_log10 = Tensor([1.0, 10.0])
        np.testing.assert_allclose(t_log10.log10().data, [0.0, 1.0])

        # var & std
        t_stat = Tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(t_stat.var().data, 2/3) # Population variance by default usually in numpy? No, functional.var calls mean((x-mean)**2), so it's population variance.
        # Let's verify numpy behavior: np.var([1,2,3]) is 0.666...
        np.testing.assert_allclose(t_stat.std().data, np.sqrt(2/3))


    def test_backward_add(self):
        x = Tensor(3.0)
        y = Tensor(4.0)
        z = x + y
        z.backward()
        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, 1.0)

    def test_backward_mul(self):
        x = Tensor(3.0)
        y = Tensor(4.0)
        z = x * y
        z.backward()
        self.assertEqual(x.grad, 4.0)
        self.assertEqual(y.grad, 3.0)

    def test_backward_complex(self):
        x = Tensor(2.0)
        y = Tensor(3.0)
        # z = (x + y) * x  = x^2 + xy = 4 + 6 = 10
        # dz/dx = 2x + y = 4 + 3 = 7
        # dz/dy = x = 2
        z = (x + y) * x
        z.backward()
        self.assertEqual(z.data, 10.0)
        self.assertEqual(x.grad, 7.0)
        self.assertEqual(y.grad, 2.0)

    def test_backward_broadcast(self):
        x = Tensor([[1., 2., 3.], [4., 5., 6.]])
        y = Tensor([1., 2., 3.])
        z = x + y
        z.sum().backward()
        
        np.testing.assert_array_equal(x.grad, np.ones_like(x.data))
        # y is added to both rows of x, so grad should be accumulated (1+1=2)
        np.testing.assert_array_equal(y.grad, np.array([2., 2., 2.]))

    def test_getitem(self):
        t = Tensor([1.0, 2.0, 3.0])
        
        # Test slice
        y = t[0]
        self.assertEqual(y.data, 1.0)
        
        y = t[1:]
        np.testing.assert_array_equal(y.data, [2.0, 3.0])

        # Test backward simple
        t.grad = None # reset
        z = t[0] * 2
        z.backward()
        np.testing.assert_array_equal(t.grad, [2.0, 0.0, 0.0])
        
        # Test backward slice
        t = Tensor([1.0, 2.0, 3.0])
        z = t[1:].sum()
        z.backward()
        np.testing.assert_array_equal(t.grad, [0.0, 1.0, 1.0])

        # Test backward with duplicate indices
        t2 = Tensor([10.0, 20.0])
        # Note: CPUBuffer delegates to numpy, so advanced indexing should work
        idx = [0, 0]
        y2 = t2[idx] # [10.0, 10.0]
        z2 = y2.sum() # 20.0
        z2.backward()
        # Gradient of sum is [1, 1] w.r.t y2.
        # y2[0] comes from t2[0], y2[1] comes from t2[0].
        # So t2[0] grad should be 1 + 1 = 2.
        np.testing.assert_array_equal(t2.grad, [2.0, 0.0])

if __name__ == '__main__':
    unittest.main()
