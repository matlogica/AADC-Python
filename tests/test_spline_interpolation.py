import numpy as np
import pytest
from scipy.interpolate import CubicSpline as ScipySpline

import aadc
from aadc.scipy.interpolate import CubicSpline


class TestNumpy:
    @staticmethod
    def _compare_splines(x, y, bc_type="not-a-knot"):
        """Helper method to compare custom spline with scipy's implementation."""
        custom_spline = CubicSpline(x, y, bc_type=bc_type)
        scipy_spline = ScipySpline(x, y, bc_type=bc_type)

        # Test at original points (vectorized)
        assert np.allclose(custom_spline(x), scipy_spline(x))

        # Test at intermediate points (vectorized)
        x_fine = np.linspace(x[0], x[-1], 1000)
        assert np.allclose(custom_spline(x_fine), scipy_spline(x_fine), rtol=1e-10, atol=1e-10)

        # Test single point evaluation
        x_single = (x[0] + x[-1]) / 2  # middle point
        assert np.allclose(custom_spline(x_single), scipy_spline(x_single), rtol=1e-10, atol=1e-10)

        # Test extrapolation below input range
        x_below = x[0] - (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
        assert np.allclose(custom_spline(x_below), scipy_spline(x_below), rtol=1e-10, atol=1e-10)

        # Test extrapolation above input range
        x_above = x[-1] + (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
        assert np.allclose(custom_spline(x_above), scipy_spline(x_above), rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_linear_case(self, bc_type):
        """Test spline interpolation on linear data."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_quadratic_case(self, bc_type):
        """Test spline interpolation on quadratic function."""
        x = np.linspace(-5, 5, 10)
        y = x**2
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("num_points", [5, 20])
    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_sine_wave(self, num_points, bc_type):
        """Test spline interpolation on sine wave with different sampling densities."""
        x = np.linspace(0, 2 * np.pi, num_points)
        y = np.sin(x)
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_exponential(self, bc_type):
        """Test spline interpolation on exponential function."""
        x = np.linspace(0, 2, 8)
        y = np.exp(x)
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_non_uniform_spacing(self, bc_type):
        """Test spline interpolation with non-uniform x spacing."""
        x = np.array([0.0, 0.1, 0.3, 0.7, 0.8, 1.0])
        y = x**3
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_large_numbers(self, bc_type):
        """Test spline interpolation with large numbers."""
        x = np.linspace(1000, 1100, 11)
        y = x**2
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_small_numbers(self, bc_type):
        """Test spline interpolation with very small numbers."""
        x = np.linspace(0.0001, 0.001, 10)
        y = np.sqrt(x)
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_oscillating_function(self, bc_type):
        """Test spline interpolation on oscillating function."""
        x = np.linspace(-2, 2, 15)
        y = np.sin(2 * x) * np.exp(-(x**2))
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_constant_function(self, bc_type):
        """Test spline interpolation on constant function."""
        x = np.linspace(0, 1, 5)
        y = np.ones_like(x)
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_step_like_function(self, bc_type):
        """Test spline interpolation on step-like function."""
        x = np.linspace(-1, 1, 21)
        y = 1.0 / (1.0 + np.exp(-20 * x))
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_single_point_evaluation(self, bc_type):
        """Test spline interpolation with single point evaluation."""
        x = np.linspace(0, 1, 5)
        y = x**2
        custom_spline = CubicSpline(x, y, bc_type=bc_type)
        scipy_spline = ScipySpline(x, y, bc_type=bc_type)

        # Test several individual points
        test_points = [0.25, 0.5, 0.75]
        for point in test_points:
            assert np.allclose(custom_spline(point), scipy_spline(point), rtol=1e-10, atol=1e-10)


class TestAADC:
    @staticmethod
    def _compare_splines(x, y, bc_type="not-a-knot", fd_eps_wrt_x=None, fd_eps_wrt_x0=None):
        """Helper method to compare custom spline with scipy's implementation."""

        with aadc.record_kernel() as kernel:
            x_active = aadc.array(x)
            x_arg = x_active.mark_as_input()

            y_active = aadc.array(y)
            y_arg = y_active.mark_as_input()

            custom_spline = CubicSpline(x_active, y_active, bc_type=bc_type)

            x0 = aadc.idouble((x[0] + x[-1]) / 2)
            x0_arg = x0.mark_as_input()
            y0 = custom_spline(x0)
            y0_out = y0.mark_as_output()

        TestAADC._test_output_value(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type)
        TestAADC._test_jac_wrt_y(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type)

        if fd_eps_wrt_x0 is not None:
            TestAADC._test_jac_wrt_x0(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, fd_eps_wrt_x0)
        else:
            TestAADC._test_jac_wrt_x0(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type)

        if fd_eps_wrt_x is not None:
            TestAADC._test_jac_wrt_x(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, fd_eps_wrt_x)
        else:
            TestAADC._test_jac_wrt_x(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type)

    @staticmethod
    def _test_output_value(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type):
        scipy_spline = ScipySpline(x, y, bc_type=bc_type)
        f = aadc.VectorFunctionWithAD(kernel, args=[x0_arg], res=[y0_out], param_args=np.r_[y_arg, x_arg])
        f.set_params(np.r_[y, x])

        # Test at original points (vectorized)
        res, _ = f.evaluate(x[:, np.newaxis])
        assert np.allclose(res.flatten(), scipy_spline(x))

        # Test at intermediate points (vectorized)
        x_fine = np.linspace(x[0], x[-1], 1000)
        res, _ = f.evaluate(x_fine[:, np.newaxis])
        assert np.allclose(res.flatten(), scipy_spline(x_fine), rtol=1e-10, atol=1e-10)

        # Test single point evaluation
        x_single = (x[0] + x[-1]) / 2  # middle point
        res, _ = f.evaluate(np.array([x_single])[np.newaxis, :])
        assert np.allclose(res.flatten(), scipy_spline(x_single), rtol=1e-10, atol=1e-10)

        # Test extrapolation below input range
        x_below = x[0] - (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
        res, _ = f.evaluate(x_below[:, np.newaxis])
        assert np.allclose(res.flatten(), scipy_spline(x_below), rtol=1e-10, atol=1e-10)

        # Test extrapolation above input range
        x_above = x[-1] + (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
        res, _ = f.evaluate(x_above[:, np.newaxis])
        assert np.allclose(res.flatten(), scipy_spline(x_above), rtol=1e-10, atol=1e-10)

    @staticmethod
    def _test_jac_wrt_x0(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, eps=1e-6):
        # Test derivatives w.r.t. evaluation points using finite differences
        scipy_spline = ScipySpline(x, y, bc_type=bc_type)
        f = aadc.VectorFunctionWithAD(kernel, args=[x0_arg], res=[y0_out], param_args=np.r_[y_arg, x_arg])
        f.set_params(np.r_[y, x])

        def compute_finite_diff(eval_points):
            """Helper function to compute finite differences at given evaluation points."""
            scipy_deriv = np.zeros_like(eval_points)
            for i in range(len(eval_points)):
                x_plus = eval_points[i] + eps
                x_minus = eval_points[i] - eps
                scipy_deriv[i] = (scipy_spline(x_plus) - scipy_spline(x_minus)) / (2 * eps)
            return scipy_deriv

        # Test at original points
        scipy_deriv = compute_finite_diff(x)
        _, jac = f.evaluate(x[:, np.newaxis])
        assert np.allclose(jac.flatten(), scipy_deriv, rtol=1e-6, atol=2e-5)

        # Test at intermediate points
        x_test = np.linspace(x[0], x[-1], 10)
        scipy_deriv = compute_finite_diff(x_test)
        _, jac = f.evaluate(x_test[:, np.newaxis])
        assert np.allclose(jac.flatten(), scipy_deriv, rtol=1e-6, atol=2e-5)

        # Test at extrapolation points
        x_below = x[0] - (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
        x_above = x[-1] + (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
        x_extra = np.concatenate([x_below, x_above])
        scipy_deriv = compute_finite_diff(x_extra)
        _, jac = f.evaluate(x_extra[:, np.newaxis])
        assert np.allclose(jac.flatten(), scipy_deriv, rtol=1e-6, atol=2e-5)

    @staticmethod
    def _test_jac_wrt_y(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type):
        # Test Jacobian w.r.t. y
        f = aadc.VectorFunctionWithAD(kernel, args=y_arg, res=[y0_out], param_args=x_arg, batch_param_args=[x0_arg])
        f.set_params(x)

        eps = 1e-2  # eps can be relatively large because we expect linear relationship
        np.random.seed(42)

        def compute_finite_diff(eval_points, y_test):
            """Helper function to compute finite differences at given evaluation points."""
            scipy_deriv = np.zeros((len(y), len(eval_points)))
            for i in range(len(x)):
                shock = np.zeros_like(y_test)
                shock[i] = eps
                scipy_spline_plus = ScipySpline(x, y_test + shock, bc_type=bc_type)
                scipy_spline_minus = ScipySpline(x, y_test - shock, bc_type=bc_type)
                scipy_deriv[i] = (scipy_spline_plus(eval_points) - scipy_spline_minus(eval_points)) / (2 * eps)
            return scipy_deriv

        # Test Jacobian with original y values and then random perturbations
        for i in range(10):
            y_test = y if i == 0 else y + np.random.randn(*y.shape)  # First iteration uses original y

            # Test at original points
            f.set_batch_params(x[:, np.newaxis])
            _, jac = f.evaluate(y_test[np.newaxis, :])
            scipy_deriv = compute_finite_diff(x, y_test)
            assert np.allclose(np.squeeze(jac).T, scipy_deriv, rtol=1e-6, atol=1e-6)

            # Test at intermediate points
            x_test = np.linspace(x[0], x[-1], 10)
            f.set_batch_params(x_test[:, np.newaxis])
            _, jac = f.evaluate(y_test[np.newaxis, :])
            scipy_deriv = compute_finite_diff(x_test, y_test)
            assert np.allclose(np.squeeze(jac).T, scipy_deriv, rtol=1e-6, atol=1e-6)

            # Test at extrapolation points
            x_below = x[0] - (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
            x_above = x[-1] + (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
            x_extra = np.concatenate([x_below, x_above])
            f.set_batch_params(x_extra[:, np.newaxis])
            _, jac = f.evaluate(y_test[np.newaxis, :])
            scipy_deriv = compute_finite_diff(x_extra, y_test)
            assert np.allclose(np.squeeze(jac).T, scipy_deriv, rtol=5e-6, atol=5e-6)

    @staticmethod
    def _test_jac_wrt_x(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, eps=1e-7):
        # Test Jacobian w.r.t. x
        f = aadc.VectorFunctionWithAD(kernel, args=x_arg, res=[y0_out], param_args=y_arg, batch_param_args=[x0_arg])
        f.set_params(y)

        np.random.seed(42)

        def compute_finite_diff(eval_points, x_test):
            """Helper function to compute finite differences at given evaluation points."""
            scipy_deriv = np.zeros((len(x), len(eval_points)))
            for i in range(len(x)):
                shock = np.zeros_like(x_test)
                shock[i] = eps
                scipy_spline_plus = ScipySpline(x_test + shock, y, bc_type=bc_type)
                scipy_spline_minus = ScipySpline(x_test - shock, y, bc_type=bc_type)
                scipy_deriv[i] = (scipy_spline_plus(eval_points) - scipy_spline_minus(eval_points)) / (2 * eps)
            return scipy_deriv

        # Test at original points
        f.set_batch_params(x[:, np.newaxis])
        _, jac = f.evaluate(x[np.newaxis, :])
        scipy_deriv = compute_finite_diff(x, x)
        assert np.allclose(np.squeeze(jac).T, scipy_deriv, rtol=1e-6, atol=1e-6)

        # Test at intermediate points
        x_eval = np.linspace(x[0], x[-1], 10)
        f.set_batch_params(x_eval[:, np.newaxis])
        _, jac = f.evaluate(x[np.newaxis, :])
        scipy_deriv = compute_finite_diff(x_eval, x)
        assert np.allclose(np.squeeze(jac).T, scipy_deriv, rtol=1e-6, atol=1e-6)

        # Test at extrapolation points
        x_below = x[0] - (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
        x_above = x[-1] + (x[-1] - x[0]) * np.array([0.1, 0.5, 1.0])
        x_extra = np.concatenate([x_below, x_above])
        f.set_batch_params(x_extra[:, np.newaxis])
        _, jac = f.evaluate(x[np.newaxis, :])
        scipy_deriv = compute_finite_diff(x_extra, x)
        assert np.allclose(np.squeeze(jac).T, scipy_deriv, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_linear_case(self, bc_type):
        """Test spline interpolation on linear data."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_quadratic_case(self, bc_type):
        """Test spline interpolation on quadratic function."""
        x = np.linspace(-5, 5, 10)
        y = x**2
        self._compare_splines(x, y, bc_type=bc_type, fd_eps_wrt_x=1e-4)

    @pytest.mark.parametrize("num_points", [5, 20])
    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_sine_wave(self, num_points, bc_type):
        """Test spline interpolation on sine wave with different sampling densities."""
        x = np.linspace(0, 2 * np.pi, num_points)
        y = np.sin(x)
        self._compare_splines(x, y, bc_type=bc_type, fd_eps_wrt_x=1e-4)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_exponential(self, bc_type):
        """Test spline interpolation on exponential function."""
        x = np.linspace(0, 2, 8)
        y = np.exp(x)
        self._compare_splines(x, y, bc_type=bc_type, fd_eps_wrt_x=1e-5)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_non_uniform_spacing(self, bc_type):
        """Test spline interpolation with non-uniform x spacing."""
        x = np.array([0.0, 0.1, 0.3, 0.7, 0.8, 1.0])
        y = x**3
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_oscillating_function(self, bc_type):
        """Test spline interpolation on oscillating function."""
        x = np.linspace(-2, 2, 15)
        y = np.sin(2 * x) * np.exp(-(x**2))
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_constant_function(self, bc_type):
        """Test spline interpolation on constant function."""
        x = np.linspace(0, 1, 5)
        y = np.ones_like(x)
        self._compare_splines(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_step_like_function(self, bc_type):
        """Test spline interpolation on step-like function."""
        x = np.linspace(-1, 1, 21)
        y = 1.0 / (1.0 + np.exp(-20 * x))
        self._compare_splines(x, y, bc_type=bc_type)
