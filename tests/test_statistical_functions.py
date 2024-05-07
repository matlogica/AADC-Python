import numpy as np
from aadc import Functions, idouble
from aadc.ndarray import AADCArray


def test_cumsum():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.cumsum(val)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert val == np.array([1.0, 3.0, 6.0])


def test_max():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.max(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 3.0


def test_mean():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.mean(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 2.0


def test_min():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.min(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 1.0


def test_prod():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.prod(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 6.0


def test_std():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.std(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert round(val, 2) == 0.82


def test_sum():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.sum(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert val == 6.0


def test_sum_2d():
    func = Functions()
    func.start_recording()
    val = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    val = np.sum(val, axis=0)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == [4.0, 6.0])


def test_sum_3d():
    func = Functions()
    func.start_recording()
    val = AADCArray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    val.mark_as_input()
    val = np.sum(val, axis=2)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[3.0, 7.0], [11.0, 15.0]]))


def test_var():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.var(val)
    func.stop_recording()
    assert isinstance(val, idouble)
    assert round(val, 2) == 0.67


def test_diff():
    func = Functions()
    func.start_recording()
    val = AADCArray([2.0, 1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.diff(val)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert val == np.array([1.0, -1.0, -4.0])
