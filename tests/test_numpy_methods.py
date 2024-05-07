import aadc
import numpy as np
import pytest


def test_all():
    func = aadc.Functions()
    func.start_recording()
    val1 = aadc.array([1.0, 2.0])
    val1.mark_as_input()
    val2 = aadc.array([3.0, 3.0])
    val2.mark_as_input()
    bool_a = val1 < val2
    result = bool_a.all()
    func.stop_recording()
    assert isinstance(result, aadc.ibool)
    assert result


def test_any():
    func = aadc.Functions()
    func.start_recording()
    val1 = aadc.array([1.0, 2.0])
    val1.mark_as_input()
    val2 = aadc.array([3.0, 1.0])
    val2.mark_as_input()
    bool_a = val1 < val2
    result = bool_a.any()
    func.stop_recording()
    assert isinstance(result, aadc.ibool)
    assert result


@pytest.mark.skip("Not supported for now")
def test_argmax():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.argmax()
    func.stop_recording()
    assert isinstance(result, aadc.idouble)
    assert result == 2


@pytest.mark.skip("Not supported for now")
def test_argmin():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([3.0, 2.0, 1.0])
    val.mark_as_input()
    result = val.argmin()
    func.stop_recording()
    assert isinstance(result, aadc.idouble)
    assert result == 2


def test_conj():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.conj()
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0])
    assert (result == expected).all()


def test_conjugate():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.conjugate()
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0])
    assert (result == expected).all()


def test_max():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.max()
    func.stop_recording()
    expected = 3
    assert result == expected


def test_mean():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.mean()
    func.stop_recording()
    expected = 2
    assert result == expected


def test_prod():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.prod()
    func.stop_recording()
    expected = 6
    assert result == expected


def test_ptp():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.ptp()
    func.stop_recording()
    expected = 2
    assert result == expected


@pytest.mark.skip("Currently not supported")
def test_ptp_multidim_with_out():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    out = aadc.array([0.0, 0.0])
    val.ptp(axis=1, out=out)
    func.stop_recording()
    expected = [2.0, 2.0]
    assert np.allclose(out, expected)


@pytest.mark.skip("Currently not supported")
def test_round():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.2, 2.3, 3.4])
    val.mark_as_input()
    result = val.round()
    func.stop_recording()
    expected = [1.0, 2.0, 3.0]
    assert np.allclose(result, expected)


def test_std():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.std()
    func.stop_recording()
    expected = np.std([1, 2, 3])
    assert result == expected


def test_sum():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.sum()
    func.stop_recording()
    expected = 6
    assert result == expected


def test_var():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    result = val.var()
    func.stop_recording()
    expected = 1.25
    assert result == expected


@pytest.mark.skip("Currently not supported")
def test_argsort():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([2.0, 1.0, 4.0, 3.0])
    val.mark_as_input()
    result = val.argsort()
    func.stop_recording()
    expected = [1, 0, 3, 2]
    assert np.array_equal(result, expected)


@pytest.mark.skip("Currently not supported")
def test_choose():
    func = aadc.Functions()
    func.start_recording()
    val1 = aadc.array([0.0, 1.0])
    val2 = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val1.mark_as_input()
    val2.mark_as_input()
    result = val1.choose(val2)
    func.stop_recording()
    expected = [3, 4]
    assert np.array_equal(result, expected)


@pytest.mark.skip("Currently not supported")
def test_compress():
    func = aadc.Functions()
    func.start_recording()
    val1 = aadc.array([0.0, 1.0, 1.0, 0.0, 1.0])
    val2 = aadc.array([10.0, 11.0, 12.0, 13.0, 14.0])
    val1.mark_as_input()
    val2.mark_as_input()
    result = val2.compress(val1)
    func.stop_recording()
    expected = [11, 12, 14]
    assert np.array_equal(result, expected)


def test_cumprod():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.cumprod()
    func.stop_recording()
    expected = [1, 2, 6]
    assert np.array_equal(result, expected)


def test_cumsum():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.cumsum()
    func.stop_recording()
    expected = [1, 3, 6]
    assert np.array_equal(result, expected)


def test_diagonal():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    result = val.diagonal()
    func.stop_recording()
    expected = [1, 4]
    assert np.array_equal(result, expected)


@pytest.mark.skip("Currently not supported")
def test_nonzero():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 0.0, 3.0])
    val.mark_as_input()
    result = val.nonzero()
    func.stop_recording()
    expected = (aadc.array([0.0, 2.0]),)
    assert np.array_equal(result, expected)


@pytest.mark.skip("Currently not supported")
def test_partition():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([5.0, 4.0, 3.0, 2.0, 1.0])
    val.mark_as_input()
    result = val.partition(3)
    func.stop_recording()
    expected = (aadc.array([3.0, 2.0, 1.0, 4.0, 5.0]),)
    assert np.array_equal(result, expected)


@pytest.mark.skip("Currently not supported")
def test_put():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    indices = [0, 1]
    values = [10, 20]
    result = val.put(indices, values)
    func.stop_recording()
    expected = aadc.array([[10.0, 2.0], [20.0, 4.0]])
    assert np.array_equal(result, expected)


def test_ravel():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    val.mark_as_input()
    result = val.ravel()
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.array_equal(result, expected)


def test_repeat():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.repeat(2)
    func.stop_recording()
    expected = aadc.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    assert np.array_equal(result, expected)


def test_resize():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    val.mark_as_input()
    result = val.resize((3, 6))
    func.stop_recording()
    assert (result.shape[0] == 3) and (result.shape[1] == 6)


@pytest.mark.skip("Currently not supported")
def test_searchsorted():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0, 4.0, 5.0])
    val.mark_as_input()
    result = val.searchsorted(3)
    func.stop_recording()
    expected = 2
    assert result == expected


def test_squeeze():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[[1.0], [2.0], [3.0]]])
    val.mark_as_input()
    result = val.squeeze()
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_clip():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.clip(1, 2)
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 2.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_min():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.min()
    func.stop_recording()
    expected = 1
    assert isinstance(expected, int)
    assert result == expected


def test_trace():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    result = val.trace()
    func.stop_recording()
    expected = 5
    assert isinstance(expected, int)
    assert result == expected


def test_copy():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.copy()
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_reshape():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    val.mark_as_input()
    result = val.reshape((2, 3))
    func.stop_recording()
    expected = aadc.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


@pytest.mark.skip("Currently not supported")
def test_sort():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([2.0, 1.0, 3.0])
    val.mark_as_input()
    result = val.sort()
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_swapaxes():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    val.mark_as_input()
    result = val.swapaxes(0, 1)
    func.stop_recording()
    expected = aadc.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_take():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0, 4.0])
    val.mark_as_input()
    indices = [0, 2]
    result = val.take(indices)
    func.stop_recording()
    expected = aadc.array([1.0, 3.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_transpose():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    result = val.transpose()
    func.stop_recording()
    expected = aadc.array([[1.0, 3.0], [2.0, 4.0]])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_astype():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.astype(float)
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


@pytest.mark.skip("Currently not supported")
def test_byteswap():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1, 2, 3], dtype="<i4")
    val.mark_as_input()
    result = val.byteswap()
    func.stop_recording()
    expected = aadc.array([67305985, 134678021, 201990457], dtype=">i4")
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_fill():
    val = aadc.array([1.0, 2.0, 3.0])
    val.fill(aadc.idouble(5.0))
    expected = aadc.array([5.0, 5.0, 5.0])
    assert isinstance(val, aadc.ndarray.AADCArray)
    assert np.array_equal(val, expected)


def test_flatten():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    result = val.flatten()
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0, 4.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_item():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([[1.0, 2.0], [3.0, 4.0]])
    val.mark_as_input()
    result = val.item(3)
    func.stop_recording()
    expected = 4
    assert result == expected


def test_itemset():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    val.itemset(1, 5.0)
    result = val
    func.stop_recording()
    expected = aadc.array([1.0, 5.0, 3.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


@pytest.mark.skip(reason="Currently not supported")
def test_newbyteorder():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1, 2, 3], dtype="<i4")
    val.mark_as_input()
    result = val.newbyteorder()
    func.stop_recording()
    expected = aadc.array([67305985, 134678021, 201990457], dtype=">i4")
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_setflags():
    val = aadc.array([1, 2, 3])
    val.setflags(write=False)
    assert val._buffer.flags["WRITEABLE"] is False


@pytest.mark.skip(reason="Currently not supported")
def test_tofile():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    val.tofile("test.dat")
    result = aadc.fromfile("test.dat")
    func.stop_recording()
    expected = aadc.array([1.0, 2.0, 3.0])
    assert isinstance(expected, aadc.ndarray.AADCArray)
    assert np.array_equal(result, expected)


def test_tolist():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.tolist()
    func.stop_recording()
    expected = [1, 2, 3]
    assert isinstance(result, list)
    assert np.allclose(aadc.array(result), expected)


@pytest.mark.skip(reason="Currently not supported")
def test_tostring():
    func = aadc.Functions()
    func.start_recording()
    val = aadc.array([1.0, 2.0, 3.0])
    val.mark_as_input()
    result = val.tostring()
    func.stop_recording()
    expected = b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"
    assert result == expected


def test_view():
    val = aadc.array([1.1, 2.1, 3.1])
    result = val.view(dtype=int)
    expected = aadc.array([4607632778762754458, 4611911198408756429, 4614162998222441677])
    assert np.array_equal(result, expected)
