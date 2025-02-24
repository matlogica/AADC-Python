import time

import jax
import jax.numpy as jnp
from jax import random

import settings

jax.config.update('jax_platform_name', 'cpu')


def simulate_paths(sigma: jnp.ndarray, rfr: jnp.ndarray, s0: jnp.ndarray, 
                  num_paths: int, num_tsteps: int, final_time: jnp.ndarray,
                  key: random.PRNGKey) -> jnp.ndarray:
    dt = final_time/num_tsteps
    gbm = jnp.zeros((num_paths, num_tsteps))
    
    gbm = gbm.at[:, 0].set(s0)
    bm_curr = jnp.zeros(num_paths)
    
    def body_fun(i, state):
        gbm, bm_curr, key = state
        key, subkey = random.split(key)
        t = i * dt
        bm_curr = bm_curr + random.normal(subkey, (num_paths,)) * jnp.sqrt(dt)
        gbm = gbm.at[:, i].set(
            s0 * jnp.exp((rfr - 0.5*sigma**2)*t + sigma*bm_curr)
        )
        return gbm, bm_curr, key

    # Use scan for the loop since JAX doesn't support traditional loops
    gbm, _, _ = jax.lax.fori_loop(
        1, num_tsteps, 
        lambda i, state: body_fun(i, state),
        (gbm, bm_curr, key)
    )
    
    return gbm

def evaluate_payoff(paths: jnp.ndarray, barrier: jnp.ndarray, 
                   strike: jnp.ndarray) -> jnp.ndarray:
    scaling_factor = 1e+1
    barrier_distances = (paths - barrier) * scaling_factor
    survival_probs = jnp.prod((1 + jnp.tanh(barrier_distances)) / 2, axis=1)
    call_payoff = jnp.maximum(paths[:, -1] - strike, 0.)
    payoffs = survival_probs * call_payoff
    return payoffs

def get_price(spot: jnp.ndarray, strike: jnp.ndarray, rfr: jnp.ndarray, 
              vol: jnp.ndarray, barrier: jnp.ndarray, expiry: jnp.ndarray,
              num_paths: int, num_tsteps: int, key: random.PRNGKey) -> jnp.ndarray:
    paths = simulate_paths(vol, rfr, spot, num_paths, num_tsteps, expiry, key)
    payoffs = evaluate_payoff(paths, barrier, strike)
    price = jnp.mean(payoffs)
    return jnp.exp(-rfr*expiry) * price

if __name__ == '__main__':
    M = settings.NUM_TIME_STEPS
    N = settings.NUM_PATHS

    spot = jnp.array(settings.SPOT)
    strike = jnp.array(settings.STRIKE)
    rfr = jnp.array(settings.RISK_FREE_RATE)
    vol = jnp.array(settings.VOLATILITY)
    barrier = jnp.array(settings.BARRIER)
    expiry = jnp.array(settings.EXPIRY)
    key = random.PRNGKey(settings.RANDOM_SEED)

    get_price_jit = jax.jit(get_price, static_argnums=(6, 7))  # only num_paths and num_tsteps are static
    get_price_and_grad = jax.value_and_grad(get_price, argnums=(0, 1, 2, 3, 4, 5))

    print("Running in JIT mode...")
    
    # First pass (compilation)
    print("First pass (compilation)...")
    start_time = time.time()
    price, grads = get_price_and_grad(spot, strike, rfr, vol, barrier, expiry, N, M, key)
    print(f"--Price: {price}")
    print("--Calculated gradients:")
    print(f"----Grad w.r.t. spot: {grads[0]}")
    print(f"----Grad w.r.t. strike: {grads[1]}")
    print(f"----Grad w.r.t. rfr: {grads[2]}")
    print(f"----Grad w.r.t. expiry: {grads[5]}")
    print(f"----Grad w.r.t. vol: {grads[3]}")
    print(f"----Grad w.r.t. barrier: {grads[4]}")
    elapsed_time = time.time() - start_time  # JAX uses lazy evaluation for gradients
    print(f"--Total execution time (1st pass: RNG + evaluate + compilation): {elapsed_time:.4f}s")

    # Second pass (after compilation)
    print("\nSecond pass...")
    start_time = time.time()
    key, subkey = random.split(key)
    price, grads = get_price_and_grad(spot, strike, rfr, vol, barrier, expiry, N, M, subkey)
    print(f"--Price: {price}")
    print("--Calculated gradients:")
    print(f"----Grad w.r.t. spot: {grads[0]}")
    print(f"----Grad w.r.t. strike: {grads[1]}")
    print(f"----Grad w.r.t. rfr: {grads[2]}")
    print(f"----Grad w.r.t. expiry: {grads[5]}")
    print(f"----Grad w.r.t. vol: {grads[3]}") 
    print(f"----Grad w.r.t. barrier: {grads[4]}")
    elapsed_time = time.time() - start_time  # JAX uses lazy evaluation for gradients
    print(f"--Total execution time (2nd pass: RNG + evaluate): {elapsed_time:.4f}s")