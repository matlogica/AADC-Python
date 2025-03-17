import numpy as np

import aadc


def test_record() -> None:
    mat = np.array([[1.0, 2.0], [-3.0, -4.0], [5.0, 6.0]])

    def analytics(x, a):
        return np.exp(-a * np.dot(mat, x))

    rec = aadc.record(analytics, np.zeros((2,)), [0.0])

    test_set = [
        ((1.0, 2.0), 0.0),
        ((1.0, 3.0), np.log(2.0)),
        ((2.0, -1.0), np.log(3.0)),
    ]

    for x, a in test_set:
        x = np.array(x)
        r = analytics(x, a)
        rec.set_params(a)

        assert np.allclose(rec.func(x), r)
        assert np.allclose(rec.jac(x), -a * (mat.T * r).T)
