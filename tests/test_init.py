import unittest

import numpy as np

import complexnn as conn


def make_initializers():
    """One of each complex initialiser, each filling a (4, 6) dense kernel"""
    return [
        conn.ComplexInit(
            kernel_size=(4, 6), input_dim=4, weight_dim=2, criterion="glorot", seed=3
        ),
        conn.IndependentFilters(kernel_size=(4, 6), input_dim=4, weight_dim=2, seed=3),
        conn.ComplexIndependentFilters(
            kernel_size=(4, 3), input_dim=4, weight_dim=2, seed=3
        ),
    ]


class TestInitializerConfig(unittest.TestCase):
    """Initialiser configs rebuild an initialiser that gives the same values"""

    shape = (4, 6)

    def test_config_round_trip(self):
        for init in make_initializers():
            with self.subTest(type(init).__name__):
                clone = type(init).from_config(init.get_config())
                self.assertEqual(clone.get_config(), init.get_config())
                np.testing.assert_array_equal(clone(self.shape), init(self.shape))


class TestSqrtInit(unittest.TestCase):
    """SqrtInit fills a shape with 1/sqrt(2), called directly or as a layer initialiser"""

    expected = 1 / np.sqrt(2)

    def test_called_directly(self):
        value = conn.SqrtInit()((4, 6))
        np.testing.assert_allclose(value.numpy(), self.expected)

    def test_as_layer_initializer(self):
        layer = conn.ComplexBN(gamma_diag_initializer=conn.SqrtInit())
        layer.build((None, 8))
        np.testing.assert_allclose(layer.gamma_rr.numpy(), self.expected)

    def test_matches_the_working_sqrt_init_function(self):
        """complexnn.bn.sqrt_init already gets this right; SqrtInit should match it"""
        from complexnn.bn import sqrt_init

        np.testing.assert_allclose(
            conn.SqrtInit()((4, 6)).numpy(), sqrt_init((4, 6)).numpy()
        )
