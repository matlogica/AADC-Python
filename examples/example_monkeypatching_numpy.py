import time
from threading import Thread

import numpy as np

import aadc.overrides
from aadc import Functions
from aadc.ndarray import AADCArray


def problematic_code(log_returns):
    asset_price_movements = np.ones((T, assets))
    for t in range(T):
        asset_price_movements[t, :] = asset_price_movements[t - 1, :] * np.exp(log_returns)
    return asset_price_movements


if __name__ == "__main__":
    rng = np.random.default_rng(1234)

    assets = 10
    T = 100

    log_returns = AADCArray(rng.standard_normal(assets))
    funcs = Functions()
    funcs.start_recording()
    log_returns.mark_as_input()

    def without_overrides():
        time.sleep(0.5)
        print("Executing thread without context manager:")
        try:
            problematic_code(log_returns)
        except Exception as e:
            print("Exception occured:", e)

    thread = Thread(target=without_overrides)
    thread.start()

    with aadc.overrides.aadc_overrides():  # Could specify a subset here
        print("Enabled overrides.")
        time.sleep(1.0)
        print("Executing within context manager:")
        asset = problematic_code(log_returns)
        print(type(asset))

    funcs.stop_recording()
    thread.join()
