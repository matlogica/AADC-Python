from dataclasses import dataclass
import time

import aadc.evaluate_wrappers
import numpy as np
import numpy.typing as npt
from mtalg.random import MultithreadedRNG

import aadc
from aadc.recording_ctx import record_kernel
import settings


@dataclass(slots=True)
class BlackScholesModel:
    sigma: float
    rfr: float
    s0: float
    
    def simulate(self, final_time: float, normal_samples: npt.NDArray[np.double]) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        num_steps = normal_samples.shape[1]
        dt = final_time / num_steps
        time = aadc.array(np.zeros(num_steps))
        gbm = aadc.array(np.zeros_like(normal_samples))
        
        gbm[:, 0] = self.s0
        bm_curr = 0.
        
        for i in range(1, num_steps):
            time[i] = i * dt
            bm_curr = bm_curr + normal_samples[:, i] * np.sqrt(dt)
            gbm[:, i] = self.s0 * np.exp((self.rfr - self.sigma**2/2) * time[i] + self.sigma * bm_curr)
            
        return time, gbm
    

@dataclass(slots=True)
class DownAndOutCallPayoff:
    barrier: float
    strike: float
    
    def evaluate(self, paths: npt.NDArray[np.double]) -> npt.NDArray[np.double] | float:
        scaling_factor = 1e+1
        barrier_distances = (paths - self.barrier) * scaling_factor
        survival_probs = np.prod((1 + np.tanh(barrier_distances)) / 2, axis=1)
        call_payoff = np.maximum(paths[:, -1] - self.strike, 0.0)
        payoffs = survival_probs * call_payoff
        return payoffs


M = settings.NUM_TIME_STEPS
N = settings.NUM_PATHS


if __name__ == '__main__':
    start_time = time.time()
    rng = MultithreadedRNG(settings.RANDOM_SEED, num_threads=12)
    normal_samples = rng.standard_normal((1, M))
    param_names = ['spot', 'strike', 'rfr', 'expiry', 'vol', 'barrier']
    params_np = aadc.array([
        settings.SPOT, 
        settings.STRIKE, 
        settings.RISK_FREE_RATE, 
        settings.EXPIRY, 
        settings.VOLATILITY, 
        settings.BARRIER
    ])

    with record_kernel() as kernel:
        params = aadc.array(params_np)
        pin = params.mark_as_input()
        active_samples = aadc.array(normal_samples)
        asin = active_samples.mark_as_input_no_diff()

        model = BlackScholesModel(params[4], params[2], params[0])
        payoff = DownAndOutCallPayoff(params[5], params[1])
        times, paths = model.simulate(params[3], active_samples)
        price_undiscounted = payoff.evaluate(paths)
        price_discounted = price_undiscounted*np.exp(-params[2]*params[3])
        pout = price_discounted.mark_as_output()

    recording_end_time = time.time()
    print(f"Recording time: {recording_end_time - start_time}")

    normal_samples_eval = rng.standard_normal((M, N))

    rng_end_time = time.time()
    print(f"RNG time: {rng_end_time - recording_end_time}")

    inputs = {
        **{param: param_value for param, param_value in zip(pin, params_np)},
        **{sample: samples_row for sample, samples_row in zip(np.squeeze(asin), normal_samples_eval)}
    }
    request = {pout.item(): pin.tolist()}

    request_build_end = time.time()
    print(f"Request build time: {request_build_end-rng_end_time}")

    workers = aadc.ThreadPool(12)
    values, derivs = aadc.evaluate(kernel, request, inputs, workers)

    price = values[pout.item()].mean()
    print(f"--Calculated price: {price}")
    print(f"--Calculated gradients:")
    for param_name, param in zip(param_names, derivs[pout.item()].values()): 
        grad = param.mean()
        print(f"----Grad w.r.t. to {param_name}: {grad}")

    evaluate_end_time = time.time()
    print(f"RNG + evaluate + compilation time: {evaluate_end_time - start_time}")
    print(f"RNG + evaluate time: {evaluate_end_time - recording_end_time}")
    print(f"Evaluate only time: {evaluate_end_time - request_build_end}")

