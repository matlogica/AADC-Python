import pickle

import aadc


def test_pickle_idouble():
    pickle.dumps(aadc.idouble(1))
