import tensorflow as tf
import time
import settings

tf.random.set_seed(settings.RANDOM_SEED)
dtype = 'float64'

class BlackScholesModel(tf.Module):

    def __init__(self, sigma: tf.Tensor, rfr: tf.Tensor, s0: tf.Tensor):
        self.sigma: tf.Tensor = sigma
        self.rfr: tf.Tensor = rfr
        self.s0: tf.Tensor = s0


    def simulate(self, num_paths: int, num_tsteps: int, final_time: float) -> tuple[tf.Tensor, tf.Tensor]:
        num_tsteps = num_tsteps
        dt = final_time/num_tsteps
        times = tf.zeros(num_tsteps, dtype=dtype)
        gbm = tf.zeros((num_paths, num_tsteps), dtype=dtype)
        
        # Initialize first values
        gbm = tf.tensor_scatter_nd_update(gbm, [[i, 0] for i in range(num_paths)], tf.fill([num_paths], self.s0))
        bm_curr = tf.zeros(num_paths, dtype=dtype)
        
        # Time stepping loop
        for i in range(1, num_tsteps):
            times = tf.tensor_scatter_nd_update(times, [[i]], [i * dt])
            bm_curr = bm_curr + tf.random.normal((num_paths,), dtype=dtype) * tf.sqrt(dt)
            curr_gbm = self.s0 * tf.exp((self.rfr - 0.5 * self.sigma**2) * (i * dt) + self.sigma * bm_curr)
            gbm = tf.tensor_scatter_nd_update(gbm, [[j, i] for j in range(num_paths)], curr_gbm)
        return gbm, times



class DownAndOutCallPayoff(tf.Module):

    def __init__(self, barrier: tf.Tensor, strike: tf.Tensor):
        self.barrier: tf.Tensor = barrier
        self.strike: tf.Tensor = strike


    def evaluate(self, paths: tf.Tensor) -> tf.Tensor:
        # Using tanh for smoother barrier transition
        scaling_factor = 1e+1  # Controls the sharpness of the transition
        barrier_distances = (paths - self.barrier) * scaling_factor
        survival_probs = tf.reduce_prod((1 + tf.tanh(barrier_distances)) / 2, axis=1)
        
        # Regular max function for the call payoff
        call_payoff = tf.maximum(paths[:, -1] - self.strike, 0.0)
        
        # Combine smoothed barrier and payoff
        payoffs = survival_probs * call_payoff
        return payoffs
    

spot = tf.Variable(settings.SPOT, dtype=dtype)
strike = tf.Variable(settings.STRIKE, dtype=dtype)
rfr = tf.Variable(settings.RISK_FREE_RATE, dtype=dtype)
vol = tf.Variable(settings.VOLATILITY, dtype=dtype)
barrier = tf.Variable(settings.BARRIER, dtype=dtype)
expiry = tf.constant(settings.EXPIRY, dtype=dtype)

PARAMS = {
    "spot": spot, 
    "strike": strike, 
    "rfr": rfr, 
    "vol": vol, 
    "barrier": barrier,
    "expiry": expiry
}

M = settings.NUM_TIME_STEPS
N = settings.NUM_PATHS


def get_price(spot, strike, rfr, vol, barrier, expiry):
    model = BlackScholesModel(vol, rfr, spot)
    payoff = DownAndOutCallPayoff(barrier, strike)
    paths, times = model.simulate(N, M, expiry)
    payoffs = payoff.evaluate(paths)

    price = tf.reduce_mean(payoffs)
    return tf.exp(-rfr*expiry) *  price


if __name__ == '__main__':
    print("Running in compiled mode - 1st pass...")

    start_time_compile = time.time()
    get_price_compiled = tf.function(get_price)

    start_time_compile_forward = time.time()
    with tf.GradientTape() as tape:
        price = get_price_compiled(**PARAMS)
    grad = tape.gradient(price, [*PARAMS.values()])

    print(f"--Calculated gradients:")
    for i, (param_name, param) in enumerate(PARAMS.items()): 
        print(f"----Grad w.r.t. to {param_name}: {grad[i]}")
        param.grad = None
    print(f"--Calculated price: {price}")

    elapsed_time_1st_pass = time.time() - start_time_compile
    print(f"--Total execution time (1st pass: RNG + evaluate + compilation): {elapsed_time_1st_pass:.4f}s")

    print("Running in compiled mode - 2nd pass...")
    start_time_2nd_pass = time.time()
    with tf.GradientTape() as tape:
        price = get_price_compiled(**PARAMS)
    grad = tape.gradient(price, [*PARAMS.values()])

    print(f"--Calculated gradients:")
    for i, (param_name, param) in enumerate(PARAMS.items()): 
        print(f"----Grad w.r.t. to {param_name}: {grad[i]}")
        param.grad = None
    print(f"--Calculated price: {price}")

    elapsed_time_2nd_pass = time.time() - start_time_2nd_pass
    print(f"--Total execution time (2nd pass: RNG + evaluate): {elapsed_time_2nd_pass:.4f}s")

