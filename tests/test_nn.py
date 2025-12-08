import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from somegrad.tensor import Tensor
from somegrad.nn import Module, Linear, BatchNorm1d, Tanh
import somegrad.functional as F

class TestNN(unittest.TestCase):

    def test_module(self):
        m = Module()
        self.assertTrue(m.training)
        self.assertEqual(m.parameters(), [])

        m.eval()
        self.assertFalse(m.training)
        m.train()
        self.assertTrue(m.training)

    def test_linear(self):
        fan_in = 5
        fan_out = 3
        linear_layer = Linear(fan_in, fan_out, bias=True)
        
        self.assertEqual(len(linear_layer.parameters()), 2) # weight and bias
        self.assertEqual(linear_layer.weight.shape, (fan_in, fan_out))
        self.assertEqual(linear_layer.bias.shape, (fan_out,))
        
        # Test forward pass
        x = Tensor(np.random.randn(10, fan_in))
        output = linear_layer(x)
        self.assertEqual(output.shape, (10, fan_out))
        
        # Test backward pass
        output.sum().backward()
        self.assertIsNotNone(linear_layer.weight.grad)
        self.assertIsNotNone(linear_layer.bias.grad)
        self.assertEqual(linear_layer.weight.grad.shape, (fan_in, fan_out))
        self.assertEqual(linear_layer.bias.grad.shape, (fan_out,))

        # Test no bias
        linear_layer_no_bias = Linear(fan_in, fan_out, bias=False)
        self.assertEqual(len(linear_layer_no_bias.parameters()), 1)
        self.assertIsNone(linear_layer_no_bias.bias)

    def test_batchnorm1d(self):
        dim = 10
        bn_layer = BatchNorm1d(dim)
        
        self.assertEqual(len(bn_layer.parameters()), 2) # gamma and beta
        self.assertEqual(bn_layer.gamma.shape, (dim,))
        self.assertEqual(bn_layer.beta.shape, (dim,))
        self.assertEqual(bn_layer.running_mean.shape, (dim,))
        self.assertEqual(bn_layer.running_var.shape, (dim,))

        # Test training mode
        x = Tensor(np.random.randn(20, dim))
        output_train = bn_layer(x)
        self.assertEqual(output_train.shape, (20, dim))

        # Check running stats update
        initial_mean = bn_layer.running_mean.data.copy()
        initial_var = bn_layer.running_var.data.copy()
        
        # Give it a new pass to update running stats
        output_train_2 = bn_layer(x)

        # running_mean and running_var should have changed
        self.assertFalse(np.array_equal(bn_layer.running_mean.data, initial_mean))
        self.assertFalse(np.array_equal(bn_layer.running_var.data, initial_var))

        # Test eval mode
        bn_layer.eval()
        output_eval = bn_layer(x)
        self.assertEqual(output_eval.shape, (20, dim))
        
        # In eval mode, running stats should not change
        mean_after_eval = bn_layer.running_mean.data.copy()
        var_after_eval = bn_layer.running_var.data.copy()
        bn_layer(x) # another pass in eval mode
        np.testing.assert_array_equal(bn_layer.running_mean.data, mean_after_eval)
        np.testing.assert_array_equal(bn_layer.running_var.data, var_after_eval)
        
        # Test backward pass in training mode
        bn_layer.train()
        output_train.sum().backward()
        self.assertIsNotNone(bn_layer.gamma.grad)
        self.assertIsNotNone(bn_layer.beta.grad)
        self.assertEqual(bn_layer.gamma.grad.shape, (dim,))
        self.assertEqual(bn_layer.beta.grad.shape, (dim,))

    def test_tanh(self):
        tanh_activation = Tanh()
        x = Tensor(np.array([-1.0, 0.0, 1.0]))
        output = tanh_activation(x)
        
        expected_output = np.tanh(x.data)
        np.testing.assert_allclose(output.data, expected_output)
        self.assertEqual(tanh_activation.parameters(), [])

if __name__ == '__main__':
    unittest.main()
