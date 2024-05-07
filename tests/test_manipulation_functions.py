import numpy as np
from aadc import Functions, idouble
from aadc.ndarray import AADCArray


class TestBroadcastArrays:
    def test_two_active_arrays(self):
        func = Functions()
        func.start_recording()
        val1 = AADCArray([1.0, 2.0])
        val1.mark_as_input()
        val2 = AADCArray([[1.0], [2.0]])
        val2.mark_as_input()
        val1, val2 = np.broadcast_arrays(val1, val2)
        func.stop_recording()
        assert isinstance(val1, AADCArray)
        assert isinstance(val2, AADCArray)
        assert np.all(val1 == np.array([[1.0, 2.0], [1.0, 2.0]]))
        assert np.all(val2 == np.array([[1.0, 1.0], [2.0, 2.0]]))

    def test_active_idouble_inactive_aadcarray(self):
        func = Functions()
        func.start_recording()
        val1 = idouble(1.0)
        val1.mark_as_input()
        val2 = AADCArray(np.array([[1.0], [2.0]]))
        val1, val2 = np.broadcast_arrays(val1, val2)
        func.stop_recording()
        assert isinstance(val1, AADCArray)
        assert isinstance(val2, AADCArray)
        assert np.all(val1 == np.array([[1.0], [1.0]]))
        assert np.all(val2 == np.array([[1.0], [2.0]]))

    def test_active_idouble_numpy_array(self):
        func = Functions()
        func.start_recording()
        val1 = idouble(1.0)
        val1.mark_as_input()
        val2 = np.array([[1.0], [2.0]])
        val1, val2 = np.broadcast_arrays(val1, val2)
        func.stop_recording()
        assert isinstance(val1, AADCArray)
        assert isinstance(val2, AADCArray)
        assert np.all(val1 == np.array([[1.0], [1.0]]))
        assert np.all(val2 == np.array([[1.0], [2.0]]))

    def test_inactive_idouble_numpy_array(self):
        func = Functions()
        func.start_recording()
        val1 = idouble(1.0)
        val2 = np.array([[1.0], [2.0]])
        val1, val2 = np.broadcast_arrays(val1, val2)
        func.stop_recording()
        assert isinstance(val1, AADCArray)
        assert isinstance(val2, AADCArray)
        assert np.all(val1 == np.array([[1.0], [1.0]]))
        assert np.all(val2 == np.array([[1.0], [2.0]]))


def test_broadcast_to():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.broadcast_to(val, (3, 3))
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))


def test_concatenate():
    func = Functions()
    func.start_recording()
    val1 = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    val2 = AADCArray([[5.0, 6.0]])
    val1.mark_as_input()
    val2.mark_as_input()
    val = np.concatenate((val1, val2), axis=0)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_concatenate_active_inactive():
    func = Functions()
    func.start_recording()
    val1 = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    val2 = AADCArray([[5.0, 6.0]])
    val1.mark_as_input()
    val = np.concatenate((val1, val2), axis=0)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_concatenate_numpy_array():
    func = Functions()
    func.start_recording()
    val1 = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    val2 = np.array([[5.0, 6.0]])
    val1.mark_as_input()
    val = np.concatenate((val1, val2), axis=0)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_expand_dims():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0])
    val.mark_as_input()
    val = np.expand_dims(val, axis=1)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[1.0], [2.0]]))


def test_flip():
    func = Functions()
    func.start_recording()
    val = AADCArray([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    val = np.flip(val)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[4.0, 3.0], [2.0, 1.0]]))


def test_moveaxis():
    func = Functions()
    func.start_recording()
    val = AADCArray(np.ones((3, 4, 5)))
    val.mark_as_input()
    val = np.moveaxis(val, 0, -1)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert val.shape == (4, 5, 3)


def test_repeat():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.repeat(val, 3)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]))


def test_reshape():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    val.mark_as_input()
    val = np.reshape(val, (3, 2))
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_roll():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.roll(val, 1)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([3.0, 1.0, 2.0]))


def test_squeeze():
    func = Functions()
    func.start_recording()
    val = AADCArray([[[1.0], [2.0], [3.0]]])
    val.mark_as_input()
    val = np.squeeze(val)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([1.0, 2.0, 3.0]))


def test_stack():
    func = Functions()
    func.start_recording()
    val1 = AADCArray([1.0, 2.0, 3.0])
    val2 = AADCArray([4.0, 5.0, 6.0])
    val1.mark_as_input()
    val2.mark_as_input()
    val = np.stack((val1, val2))
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_tile():
    func = Functions()
    func.start_recording()
    val = AADCArray([1.0, 2.0, 3.0])
    val.mark_as_input()
    val = np.tile(val, 2)
    func.stop_recording()
    assert isinstance(val, AADCArray)
    assert np.all(val == np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))
