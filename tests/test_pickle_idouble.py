import pickle

import aadc


def test_pickle_idouble() -> None:
    pickle.dumps(aadc.idouble(1))
