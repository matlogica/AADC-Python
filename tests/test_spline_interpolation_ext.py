import pickle

import numpy as np
import pytest
from scipy.interpolate import CubicSpline as ScipySpline

import aadc
from aadc.scipy.interpolate import CubicSpline


class TestAADCExtended:
    """Extended test suite for AADC spline interpolation."""

    @staticmethod
    def _test_spline_derivatives(x, y, bc_type="not-a-knot", fd_eps_wrt_x=1e-7, fd_eps_wrt_y=1e-7, fd_eps_wrt_x0=1e-6):
        """Test spline derivatives using AADC and compare with finite differences approach."""
        # Record the AADC kernel
        x_rec = x + 10.0 # Use different values during recording
        y_rec = y + 10.0

        with aadc.record_kernel() as kernel:
            # Convert inputs to AADC arrays and mark as inputs
            x_active = aadc.array(x_rec)
            x_arg = x_active.mark_as_input()

            y_active = aadc.array(y_rec)
            y_arg = y_active.mark_as_input()

            # Create spline
            custom_spline = CubicSpline(x_active, y_active, bc_type=bc_type)

            # Evaluate at a point
            x0 = aadc.idouble((x[0] + x[-1]) / 2)  # Middle point
            x0_arg = x0.mark_as_input()
            y0 = custom_spline(x0)
            y0_out = y0.mark_as_output()

        pickled_kernel = pickle.dumps(kernel)
        kernel = pickle.loads(pickled_kernel)

        # Test output value
        TestAADCExtended._verify_output_value(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type)

        # Test Jacobian w.r.t. y values
        TestAADCExtended._verify_jacobian_wrt_y(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, fd_eps_wrt_y)

        # Test Jacobian w.r.t. x values
        TestAADCExtended._verify_jacobian_wrt_x(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, fd_eps_wrt_x)

        # Test Jacobian w.r.t. evaluation point
        TestAADCExtended._verify_jacobian_wrt_x0(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, fd_eps_wrt_x0)

    @staticmethod
    def _verify_output_value(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type):
        """Verify spline output values match between AADC and SciPy implementations."""
        # Create SciPy spline for comparison
        scipy_spline = ScipySpline(x, y, bc_type=bc_type)

        # Create vector function with AADC
        f = aadc.VectorFunctionWithAD(kernel, args=[x0_arg], res=[y0_out], param_args=np.r_[y_arg, x_arg])
        f.set_params(np.r_[y, x])

        # Test single point evaluation
        x_single = (x[0] + x[-1]) / 2
        res, _ = f.evaluate(np.array([x_single])[np.newaxis, :])
        assert np.allclose(res.flatten(), scipy_spline(x_single), rtol=1e-10, atol=1e-10)

        # Test multiple points within range
        x_fine = np.linspace(x[0], x[-1], 20)
        res, _ = f.evaluate(x_fine[:, np.newaxis])
        assert np.allclose(res.flatten(), scipy_spline(x_fine), rtol=1e-10, atol=1e-10)

        # Test extrapolation
        x_extra = np.array([x[0] - 0.5, x[-1] + 0.5])
        res, _ = f.evaluate(x_extra[:, np.newaxis])
        assert np.allclose(res.flatten(), scipy_spline(x_extra), rtol=1e-10, atol=1e-10)

    @staticmethod
    def _verify_jacobian_wrt_y(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, eps=1e-7):
        """Verify Jacobian w.r.t. y values using finite differences."""
        # Create vector function with AADC
        f = aadc.VectorFunctionWithAD(kernel, args=y_arg, res=[y0_out], param_args=x_arg, batch_param_args=[x0_arg])
        f.set_params(x)

        # Helper function for finite differences
        def compute_fd_jacobian(eval_points, y_values):
            """Compute Jacobian using finite differences."""
            n_y = len(y_values)
            n_points = len(eval_points)
            jac = np.zeros((n_y, n_points))

            for i in range(n_y):
                y_plus = y_values.copy()
                y_plus[i] += eps

                y_minus = y_values.copy()
                y_minus[i] -= eps

                spline_plus = ScipySpline(x, y_plus, bc_type=bc_type)
                spline_minus = ScipySpline(x, y_minus, bc_type=bc_type)

                jac[i, :] = (spline_plus(eval_points) - spline_minus(eval_points)) / (2 * eps)

            return jac

        # Test at different points
        test_points = [
            np.array([x[0] + 0.1]),  # Near start
            np.array([(x[0] + x[-1]) / 2]),  # Middle
            np.array([x[-1] - 0.1]),  # Near end
            np.linspace(x[0], x[-1], 5),  # Multiple points in range
            np.array([x[0] - 0.2, x[-1] + 0.2])  # Extrapolation
        ]

        for points in test_points:
            f.set_batch_params(points[:, np.newaxis])
            _, aadc_jac = f.evaluate(y[np.newaxis, :])
            fd_jac = compute_fd_jacobian(points, y)

            assert np.allclose(np.squeeze(aadc_jac).T, np.squeeze(fd_jac), rtol=1e-5, atol=1e-5)

    @staticmethod
    def _verify_jacobian_wrt_x(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, eps=1e-7):
        """Verify Jacobian w.r.t. x knot points using finite differences."""
        # Create vector function with AADC
        f = aadc.VectorFunctionWithAD(kernel, args=x_arg, res=[y0_out], param_args=y_arg, batch_param_args=[x0_arg])
        f.set_params(y)

        # Helper function for finite differences
        def compute_fd_jacobian(eval_points, x_values):
            """Compute Jacobian using finite differences."""
            n_x = len(x_values)
            n_points = len(eval_points)
            jac = np.zeros((n_x, n_points))

            for i in range(n_x):
                x_plus = x_values.copy()
                x_plus[i] += eps

                x_minus = x_values.copy()
                x_minus[i] -= eps

                spline_plus = ScipySpline(x_plus, y, bc_type=bc_type)
                spline_minus = ScipySpline(x_minus, y, bc_type=bc_type)

                jac[i, :] = (spline_plus(eval_points) - spline_minus(eval_points)) / (2 * eps)

            return jac

        # Test at different points
        test_points = [
            np.array([x[0] + 0.1]),  # Near start
            np.array([(x[0] + x[-1]) / 2]),  # Middle
            np.array([x[-1] - 0.1]),  # Near end
            np.linspace(x[0], x[-1], 5),  # Multiple points in range
            np.array([x[0] - 0.2, x[-1] + 0.2])  # Extrapolation
        ]

        for points in test_points:
            f.set_batch_params(points[:, np.newaxis])
            _, aadc_jac = f.evaluate(x[np.newaxis, :])
            fd_jac = compute_fd_jacobian(points, x)

            assert np.allclose(np.squeeze(aadc_jac).T, np.squeeze(fd_jac), rtol=1e-5, atol=1e-5)

    @staticmethod
    def _verify_jacobian_wrt_x0(x, y, kernel, x_arg, y_arg, x0_arg, y0_out, bc_type, eps=1e-6):
        """Verify Jacobian w.r.t. evaluation point using finite differences."""
        # Create SciPy spline for comparison
        scipy_spline = ScipySpline(x, y, bc_type=bc_type)

        # Create vector function with AADC
        f = aadc.VectorFunctionWithAD(kernel, args=[x0_arg], res=[y0_out], param_args=np.r_[y_arg, x_arg])
        f.set_params(np.r_[y, x])

        # Helper function for finite differences
        def compute_fd_derivative(eval_points):
            """Compute derivative using finite differences."""
            derivs = np.zeros_like(eval_points)

            for i in range(len(eval_points)):
                x_plus = eval_points[i] + eps
                x_minus = eval_points[i] - eps

                derivs[i] = (scipy_spline(x_plus) - scipy_spline(x_minus)) / (2 * eps)

            return derivs

        # Test at different points
        test_points = [
            np.array([x[0] + 0.1]),  # Near start
            np.array([(x[0] + x[-1]) / 2]),  # Middle
            np.array([x[-1] - 0.1]),  # Near end
            np.linspace(x[0], x[-1], 5),  # Multiple points in range
            np.array([x[0] - 0.2, x[-1] + 0.2])  # Extrapolation
        ]

        for points in test_points:
            _, aadc_jac = f.evaluate(points[:, np.newaxis])
            fd_derivs = compute_fd_derivative(points)

            assert np.allclose(aadc_jac.flatten(), fd_derivs, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_cubic_polynomial(self, bc_type):
        """Test spline on cubic polynomial data."""
        x = np.linspace(-3, 3, 7)
        y = x**3 - 2*x**2 + 3*x - 1
        self._test_spline_derivatives(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_rational_function(self, bc_type):
        """Test spline on rational function."""
        x = np.linspace(1, 5, 9)
        y = 1 / (1 + x**2)
        self._test_spline_derivatives(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_piecewise_function(self, bc_type):
        """Test spline on piecewise function."""
        x = np.linspace(-2, 2, 11)
        y = np.where(x < 0, x**2, np.sqrt(np.abs(x)))
        self._test_spline_derivatives(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_irregular_spacing(self, bc_type):
        """Test spline with irregularly spaced x points."""
        # Create irregular spacing
        x = np.array([0.0, 0.05, 0.1, 0.3, 0.35, 0.37, 0.4, 0.5, 0.8, 1.0])
        y = np.exp(-x) * np.sin(2 * np.pi * x)
        self._test_spline_derivatives(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_noise_data(self, bc_type):
        """Test spline with noisy data."""
        np.random.seed(42)
        x = np.linspace(0, 1, 10)
        base_y = np.sin(2 * np.pi * x)
        noise = np.random.normal(0, 0.05, size=len(x))
        y = base_y + noise
        self._test_spline_derivatives(x, y, bc_type=bc_type)

    @pytest.mark.parametrize("bc_type", ["natural", "clamped", "not-a-knot"])
    def test_vector_evaluation(self, bc_type):
        """Test vectorized spline evaluation and derivatives."""
        # Record the AADC kernel with vector evaluation
        x = np.linspace(0, 1, 5)
        y = x**2

        with aadc.record_kernel() as kernel:
            x_active = aadc.array(x + 10.0)
            x_arg = x_active.mark_as_input()

            y_active = aadc.array(y + 10.0)
            y_arg = y_active.mark_as_input()

            custom_spline = CubicSpline(x_active, y_active, bc_type=bc_type)

            # Vector of evaluation points
            x_eval = aadc.array([0.25, 0.5, 0.75]) + 10.0
            x_eval_arg = x_eval.mark_as_input()

            y_eval = custom_spline(x_eval)
            y_eval_out = y_eval.mark_as_output()

        pickled_kernel = pickle.dumps(kernel)
        kernel = pickle.loads(pickled_kernel)

        # Create SciPy spline for comparison
        scipy_spline = ScipySpline(x, y, bc_type=bc_type)

        # Set up AADC function
        f = aadc.VectorFunctionWithAD(kernel, args=list(x_eval_arg), res=list(y_eval_out), param_args=list(np.r_[y_arg, x_arg]))
        f.set_params(np.r_[y, x])

        # Test evaluation at multiple points
        eval_points = np.array([[0.25, 0.5, 0.75]])
        res, _ = f.evaluate(eval_points)

        expected = scipy_spline(np.array([0.25, 0.5, 0.75]))
        assert np.allclose(res.flatten(), expected, rtol=1e-10, atol=1e-10)

    def test_batch_evaluation(self):
        """Test batch evaluation of splines with different parameters."""
        # Set up multiple sets of x, y data
        x_base = np.linspace(0, 1, 5)
        batch_size = 3

        # Create batches with different y values
        y_batches = np.zeros((batch_size, len(x_base)))
        for i in range(batch_size):
            y_batches[i] = x_base**(i+1)

        # Record the AADC kernel
        with aadc.record_kernel() as kernel:
            x_active = aadc.array(x_base + 10.0)
            x_arg = x_active.mark_as_input()

            # Create a batch of y values
            y_active = aadc.array(np.zeros_like(y_batches[0]))  # Placeholder
            y_arg = y_active.mark_as_input()

            # Create spline
            custom_spline = CubicSpline(x_active, y_active)

            # Evaluate at a point
            x0 = aadc.idouble(0.5 + 10.0)  # Middle point
            x0_arg = x0.mark_as_input()
            y0 = custom_spline(x0)
            y0_out = y0.mark_as_output()

        pickled_kernel = pickle.dumps(kernel)
        kernel = pickle.loads(pickled_kernel)

        # Set up AADC function
        f = aadc.VectorFunctionWithAD(kernel, args=y_arg, res=[y0_out], param_args=np.r_[x_arg, x0_arg])
        f.set_params(np.r_[x_base, 0.5])

        # Evaluate for each batch
        results = []
        for i in range(batch_size):
            res, _ = f.evaluate(y_batches[i:i+1])

            # Compare with SciPy
            scipy_spline = ScipySpline(x_base, y_batches[i])
            expected = scipy_spline(0.5)

            assert np.allclose(res.item(), expected, rtol=1e-10, atol=1e-10)
            results.append(res.item())

        # Results should be different for each batch
        assert len(set(results)) == batch_size

    def test_multi_point_derivatives(self):
        """Test derivatives at multiple evaluation points simultaneously."""
        x = np.linspace(0, 1, 5)
        y = x**2

        # Create eval points and expected derivatives
        eval_points = np.linspace(0.1, 0.9, 9)

        # Record the AADC kernel
        with aadc.record_kernel() as kernel:
            x_active = aadc.array(x + 10.0)
            x_arg = x_active.mark_as_input()

            y_active = aadc.array(y + 10.0)
            y_arg = y_active.mark_as_input()

            # Create spline
            custom_spline = CubicSpline(x_active, y_active)

            # Multiple evaluation points
            x_evals = aadc.array(eval_points + 10.0)
            x_evals_arg = x_evals.mark_as_input()

            y_evals = custom_spline(x_evals)
            y_evals_out = y_evals.mark_as_output()

        pickled_kernel = pickle.dumps(kernel)
        kernel = pickle.loads(pickled_kernel)

        # Test Jacobian w.r.t. evaluation points
        scipy_spline = ScipySpline(x, y)

        # Create vector function with AADC
        f = aadc.VectorFunctionWithAD(kernel, args=list(x_evals_arg), res=list(y_evals_out), param_args=np.r_[y_arg, x_arg])
        f.set_params(np.r_[y, x])

        # Evaluate and get Jacobian
        _, jac = f.evaluate(eval_points[np.newaxis, :])

        # Compute finite difference for comparison
        eps = 1e-6
        fd_jac = np.zeros((len(eval_points), len(eval_points)))
        for i in range(len(eval_points)):
            x_plus = eval_points.copy()
            x_plus[i] += eps

            x_minus = eval_points.copy()
            x_minus[i] -= eps

            fd_jac[i] = (scipy_spline(x_plus) - scipy_spline(x_minus)) / (2 * eps)

        # Jacobian should be diagonal for independent evaluation points
        assert np.allclose(np.diag(jac.squeeze()), np.diag(fd_jac), rtol=1e-5, atol=1e-5)


class TestAADCPerformance:
    """Test performance and advanced usage of AADC spline interpolation."""

    def test_large_data(self):
        """Test with large number of data points."""
        # Generate large data set
        n_points = 100
        x = np.linspace(0, 10, n_points)
        y = np.sin(x) + 0.1 * np.random.randn(n_points)

        # Record the AADC kernel
        with aadc.record_kernel() as kernel:
            x_active = aadc.array(x + 10.0)
            x_arg = x_active.mark_as_input()

            y_active = aadc.array(y + 10.0)
            y_arg = y_active.mark_as_input()

            # Create spline
            custom_spline = CubicSpline(x_active, y_active)

            # Single evaluation point
            x0 = aadc.idouble(5.0 + 10.0)
            x0_arg = x0.mark_as_input()
            y0 = custom_spline(x0)
            y0_out = y0.mark_as_output()

        pickled_kernel = pickle.dumps(kernel)
        kernel = pickle.loads(pickled_kernel)

        # Test function output
        scipy_spline = ScipySpline(x, y)
        expected = scipy_spline(5.0)

        f = aadc.VectorFunctionWithAD(kernel, args=list([x0_arg]), res=list([y0_out]), param_args=np.r_[y_arg, x_arg])
        f.set_params(np.r_[y, x])

        res, _ = f.evaluate(np.array([[5.0]]))
        assert np.allclose(res.item(), expected, rtol=1e-10, atol=1e-10)

        # Test Jacobian w.r.t. y
        f = aadc.VectorFunctionWithAD(kernel, args=y_arg, res=list([y0_out]), param_args=np.r_[x_arg, x0_arg])
        f.set_params(np.r_[x, 5.0])

        _, jac = f.evaluate(y[np.newaxis, :])

        # Compute finite difference for comparison (only test a few points for performance)
        eps = 1e-7
        test_indices = [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]

        for idx in test_indices:
            y_plus = y.copy()
            y_plus[idx] += eps

            y_minus = y.copy()
            y_minus[idx] -= eps

            scipy_plus = ScipySpline(x, y_plus)
            scipy_minus = ScipySpline(x, y_minus)

            fd_deriv = (scipy_plus(5.0) - scipy_minus(5.0)) / (2 * eps)
            assert np.allclose(jac[0, 0][idx], fd_deriv, rtol=1e-5, atol=1e-5)

    def test_multiple_splines(self):
        """Test using multiple splines in the same kernel."""
        x1 = np.linspace(0, 1, 5)
        y1 = x1**2

        x2 = np.linspace(-1, 1, 7)
        y2 = np.exp(x2)

        # Record the AADC kernel
        with aadc.record_kernel() as kernel:
            # First spline
            x1_active = aadc.array(x1 + 10.0)
            x1_arg = x1_active.mark_as_input()

            y1_active = aadc.array(y1 + 10.0)
            y1_arg = y1_active.mark_as_input()

            spline1 = CubicSpline(x1_active, y1_active)

            # Second spline
            x2_active = aadc.array(x2 + 10.0)
            x2_arg = x2_active.mark_as_input()

            y2_active = aadc.array(y2 + 10.0)
            y2_arg = y2_active.mark_as_input()

            spline2 = CubicSpline(x2_active, y2_active)

            # Evaluate both splines
            x0 = aadc.idouble(0.5 + 10.0)
            x0_arg = x0.mark_as_input()

            y1_eval = spline1(x0)
            y1_out = y1_eval.mark_as_output()

            y2_eval = spline2(x0)
            y2_out = y2_eval.mark_as_output()

        pickled_kernel = pickle.dumps(kernel)
        kernel = pickle.loads(pickled_kernel)

        # Test both splines
        scipy_spline1 = ScipySpline(x1, y1)
        scipy_spline2 = ScipySpline(x2, y2)

        expected1 = scipy_spline1(0.5)
        expected2 = scipy_spline2(0.5)

        # Evaluate first spline
        f1 = aadc.VectorFunctionWithAD(kernel, args=[x0_arg], res=[y1_out],
                                      param_args=np.r_[y1_arg, x1_arg, y2_arg, x2_arg])
        f1.set_params(np.r_[y1, x1, y2, x2])

        res1, _ = f1.evaluate(np.array([[0.5]]))
        assert np.allclose(res1.item(), expected1, rtol=1e-10, atol=1e-10)

        # Evaluate second spline
        f2 = aadc.VectorFunctionWithAD(kernel, args=[x0_arg], res=[y2_out],
                                      param_args=np.r_[y1_arg, x1_arg, y2_arg, x2_arg])
        f2.set_params(np.r_[y1, x1, y2, x2])

        res2, _ = f2.evaluate(np.array([[0.5]]))
        assert np.allclose(res2.item(), expected2, rtol=1e-10, atol=1e-10)

        # Test cross-derivatives (changes in y1 should not affect spline2)
        f_cross = aadc.VectorFunctionWithAD(kernel, args=y1_arg, res=[y2_out],
                                           param_args=np.r_[x1_arg, y2_arg, x2_arg, x0_arg])
        f_cross.set_params(np.r_[x1, y2, x2, 0.5])

        _, jac = f_cross.evaluate(y1[np.newaxis, :])
        assert np.allclose(jac, 0, atol=1e-10)  # Should be zero everywhere

    def test_composition(self):
        """Test composition of splines with other operations."""
        x = np.linspace(0, 1, 5)
        y = x**2

        # Record the AADC kernel
        with aadc.record_kernel() as kernel:
            x_active = aadc.array(x + 10.0)
            x_arg = x_active.mark_as_input()

            y_active = aadc.array(y + 10.0)
            y_arg = y_active.mark_as_input()

            # Create spline
            spline = CubicSpline(x_active, y_active)

            # Evaluate at a point and apply composition of operations
            x0 = aadc.idouble(0.5 + 10.0)
            x0_arg = x0.mark_as_input()

            y0 = spline(x0)
            result = y0**2 + 3*y0 + 1  # Quadratic function of spline output
            result_out = result.mark_as_output()

        pickled_kernel = pickle.dumps(kernel)
        kernel = pickle.loads(pickled_kernel)

        # Compare with manual computation
        scipy_spline = ScipySpline(x, y)
        y0_expected = scipy_spline(0.5)
        expected = y0_expected**2 + 3*y0_expected + 1

        # Test output
        f = aadc.VectorFunctionWithAD(kernel, args=[x0_arg], res=[result_out], param_args=np.r_[y_arg, x_arg])
        f.set_params(np.r_[y, x])

        res, jac = f.evaluate(np.array([[0.5]]))
        assert np.allclose(res.item(), expected, rtol=1e-10, atol=1e-10)

        # Test derivative using chain rule
        # d/dx0 (f(g(x0))) = f'(g(x0)) * g'(x0)
        # where g(x0) = spline(x0) and f(z) = z^2 + 3z + 1
        spline_deriv = scipy_spline(0.5, 1)  # First derivative
        outer_deriv = 2*y0_expected + 3  # Derivative of the quadratic function
        expected_deriv = outer_deriv * spline_deriv

        assert np.allclose(jac.item(), expected_deriv, rtol=1e-5, atol=1e-5)

#t = TestAADCPerformance()

#t.test_cubic_polynomial("natural")
#t.test_multi_point_derivatives()
#t.test_large_data()
