import time

import torch

import settings

torch.set_default_dtype(torch.double)
torch.set_default_device('cpu')
torch.manual_seed(settings.RANDOM_SEED)

@torch.jit.script
def simulate_paths(sigma: torch.Tensor, rfr: torch.Tensor, s0: torch.Tensor, 
                  num_paths: int, num_tsteps: int, final_time: torch.Tensor) -> torch.Tensor:
    dt = final_time/num_tsteps
    gbm = torch.zeros((num_paths, num_tsteps))
    
    gbm[:, 0] = s0
    bm_curr = torch.zeros(num_paths)
    
    for i in range(1, num_tsteps):
        t = i * dt
        bm_curr = bm_curr + torch.randn(num_paths) * torch.sqrt(dt)
        gbm[:, i] = s0 * torch.exp((rfr - 0.5*sigma**2)*t + sigma*bm_curr)
        
    return gbm

@torch.jit.script
def evaluate_payoff(paths: torch.Tensor, barrier: torch.Tensor, 
                   strike: torch.Tensor) -> torch.Tensor:
    scaling_factor = 1e+1
    barrier_distances = (paths - barrier) * scaling_factor
    survival_probs = torch.prod((1 + torch.tanh(barrier_distances)) / 2, dim=1)
    call_payoff = torch.maximum(paths[:, -1] - strike, torch.tensor(0.))
    payoffs = survival_probs * call_payoff
    return payoffs

@torch.jit.script
def get_price(spot: torch.Tensor, strike: torch.Tensor, rfr: torch.Tensor, 
              vol: torch.Tensor, barrier: torch.Tensor, expiry: torch.Tensor,
              num_paths: int, num_tsteps: int) -> torch.Tensor:
    paths = simulate_paths(vol, rfr, spot, num_paths, num_tsteps, expiry)
    payoffs = evaluate_payoff(paths, barrier, strike)
    price = torch.mean(payoffs)
    return torch.exp(-rfr*expiry) * price

M = settings.NUM_TIME_STEPS
N = settings.NUM_PATHS

# Parameters
spot = torch.tensor(settings.SPOT, requires_grad=True)
strike = torch.tensor(settings.STRIKE, requires_grad=True)
rfr = torch.tensor(settings.RISK_FREE_RATE, requires_grad=True)
vol = torch.tensor(settings.VOLATILITY, requires_grad=True)
barrier = torch.tensor(settings.BARRIER, requires_grad=True)
expiry = torch.tensor(settings.EXPIRY, requires_grad=True)

PARAMS = {
    "spot": spot, 
    "strike": strike, 
    "rfr": rfr, 
    "vol": vol, 
}

if __name__ == '__main__':
    M = settings.NUM_TIME_STEPS
    N = settings.NUM_PATHS

    spot = torch.tensor(settings.SPOT, requires_grad=True)
    strike = torch.tensor(settings.STRIKE, requires_grad=True)
    rfr = torch.tensor(settings.RISK_FREE_RATE, requires_grad=True)
    vol = torch.tensor(settings.VOLATILITY, requires_grad=True)
    barrier = torch.tensor(settings.BARRIER, requires_grad=True)
    expiry = torch.tensor(settings.EXPIRY, requires_grad=True)

    print("Running in JIT mode...")
    
    print("First pass (compilation)...")
    start_time = time.time()
    price = get_price(spot, strike, rfr, vol, barrier, expiry, N, M)
    price.backward()
    elapsed_time = time.time() - start_time
    print(f"--Price: {price.item()}")
    print("--Calculated gradients:")
    print(f"----Grad w.r.t. spot: {spot.grad.item()}")
    print(f"----Grad w.r.t. strike: {strike.grad.item()}")
    print(f"----Grad w.r.t. rfr: {rfr.grad.item()}")
    print(f"----Grad w.r.t. expiry: {expiry.grad.item()}")
    print(f"----Grad w.r.t. vol: {vol.grad.item()}")
    print(f"----Grad w.r.t. barrier: {barrier.grad.item()}")
    print(f"--Total execution time (1st pass: RNG + evaluate + compilation): {elapsed_time:.4f}s")

    # Reset gradients
    for param in [spot, strike, rfr, vol, barrier, expiry]:
        param.grad = None

    # Second pass
    print("\nSecond pass...")
    start_time = time.time()
    price = get_price(spot, strike, rfr, vol, barrier, expiry, N, M)
    price.backward()
    elapsed_time = time.time() - start_time
    print(f"--Price: {price.item()}")
    print("--Calculated gradients:")
    print(f"----Grad w.r.t. spot: {spot.grad.item()}")
    print(f"----Grad w.r.t. strike: {strike.grad.item()}")
    print(f"----Grad w.r.t. rfr: {rfr.grad.item()}")
    print(f"----Grad w.r.t. expiry: {expiry.grad.item()}")
    print(f"----Grad w.r.t. vol: {vol.grad.item()}")
    print(f"----Grad w.r.t. barrier: {barrier.grad.item()}")
    print(f"--Total execution time (2nd pass: RNG + evaluate): {elapsed_time:.4f}s")