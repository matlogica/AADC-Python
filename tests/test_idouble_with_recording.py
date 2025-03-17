import numpy as np

from aadc import Functions, idouble
from aadc.ndarray import AADCArray


def test_add_array_to_idouble() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = val + np.array([3.0, 4.0])

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([7.0, 8.0])), "Result should match expected values"


def test_add_idouble_to_array() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = np.array([3.0, 4.0])
    result += val

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([7.0, 8.0])), "Result should match expected values"


def test_sub_idouble_from_array() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = np.array([3.0, 4.0])
    result -= val

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([-1.0, 0.0])), "Result should match expected values"


def test_sub_array_from_idouble() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = val - np.array([3.0, 4.0])

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([1.0, 0.0])), "Result should match expected values"


def test_mul_idouble_with_array() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = val * np.array([3.0, 4.0])

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([12.0, 16.0])), "Result should match expected values"


def test_div_idouble_by_array() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = val / np.array([3.0, 4.0])

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([4.0 / 3.0, 1.0])), "Result should match expected values"


def test_eq_idouble_with_array() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = val == np.array([3.0, 4.0])

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([False, True])), "Result should match expected values"


def test_gt_idouble_with_array() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = val > np.array([3.0, 4.0])

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([True, False])), "Result should match expected values"


def test_le_idouble_with_array() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = val <= np.array([3.0, 4.0])

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([False, True])), "Result should match expected values"


def test_ge_idouble_with_array() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = val >= np.array([3.0, 4.0])

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([True, True])), "Result should match expected values"


def test_mul_array_with_idouble() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = np.array([3.0, 4.0]) * val

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([12.0, 16.0])), "Result should match expected values"


def test_div_array_by_idouble() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = np.array([3.0, 4.0]) / val

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([3.0 / 4.0, 1.0])), "Result should match expected values"


def test_eq_array_with_idouble() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = np.array([3.0, 4.0]) == val

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([False, True])), "Result should match expected values"


def test_gt_array_with_idouble() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = np.array([3.0, 4.0]) > val

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([False, False])), "Result should match expected values"


def test_le_array_with_idouble() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = np.array([3.0, 4.0]) <= val

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([True, True])), "Result should match expected values"


def test_ge_array_with_idouble() -> None:
    func = Functions()
    func.start_recording()

    val = idouble(4.0)
    val.mark_as_input()
    result = np.array([3.0, 4.0]) >= val

    func.stop_recording()

    assert type(result) == AADCArray, "Result should be an AADCArray"
    assert np.all(result == np.array([False, True])), "Result should match expected values"
