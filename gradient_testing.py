import multiprocessing as mp
import numpy as np # Just utility
from scipy.optimize import check_grad # For comparison gradient
import SamplingMesh

def parabola(x):
    return np.sum(x**2, axis=-1)

def centered_gradient(func, xs, step_size=1e-8):
    grad = np.zeros_like(xs)
    ndim = xs.shape[-1]
    for i in range(ndim):
        delta = np.zeros((1, ndim))
        delta[0, i] = step_size / 2
        grad[:,i] = (func(xs + delta) - func(xs - delta)) / step_size
    return grad

if __name__ == '__main__':
    #SamplingMesh.DEBUG = True
    
    # First: test if our gradient does indeed approach the true gradient
    # and check associated error estimates
    
    print("Testing error estimates on gradient")
    
    # Parabolas are easy to test for, because the MSE of the gradient doesn't
    # depend on location
    
    tests = [("1D parabola", 1, parabola, np.ones(1)),
             ("2D parabola", 2, parabola, np.ones(2)),
             ("3D parabola", 3, parabola, np.ones(3))]
    
    for name, ndim, f, resolution in tests:
        SamplingMesh.NDIM = ndim
        for atol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
            with mp.Pool() as pool:
                mesh = SamplingMesh.SamplingMesh(f, pool, scale=resolution, atol=atol, rtol=0)
                xs = np.random.normal(size = (100, SamplingMesh.NDIM))
                try:
                    grads = mesh.multi_gradient(xs)
                except Exception as e:
                    print(f"Encountered error on {name} with atol {atol}")
                    raise e
            
            mse = np.mean((grads - 2 * xs)**2)
            # MSE of gradient in one dimension is curvature times 2.5 * RMSE of values
            # curvature is 2 for 1d parabola
            # nd parabola is a sum of orthogonal 1d parabolas, so the MSE is the mean of the MSEs
            if mse  > 5 * atol:
                print(f"WARNING: MSE of {mse} exceeds target of {5 * atol} for {name}.")
            else:
                print(f"MSE of {mse} within {5 * atol} for {name}.")
                
    
    # Second: check that the gradient is, in fact, *exactly* the gradient of
    # the interpolation
    
    print("Checking that gradient matches the numerical gradient of the interpolation")
    
    #SamplingMesh.DEBUG = True
    
    
    
    tests = [("1D_parabola", 1, parabola, np.ones(1)),
             ("2D_parabola", 2, parabola, np.ones(2)),
             ("3D_parabola", 3, parabola, np.ones(3))]
    
    
    for name, ndim, f, resolution in tests:
        xs = np.random.normal(size = (1000, SamplingMesh.NDIM))
        # Find rounding error mse
        inherent_rmse = np.sqrt(np.mean((2 * xs - centered_gradient(f, xs))**2))
        SamplingMesh.NDIM = ndim
        for atol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
            with mp.Pool() as pool:
                mesh = SamplingMesh.SamplingMesh(f, pool, scale=resolution, atol=atol, rtol=0)
                xs = np.random.normal(size = (1000, SamplingMesh.NDIM))
                try:
                    grads = mesh.multi_gradient(xs)
                    rmse = np.sqrt(np.mean((centered_gradient(mesh.multi_interpolate, xs) - grads)**2))
                except Exception as e:
                    print(f"Encountered error on {name} with atol {atol}")
                    raise e
            
            # We can ignore anything that's the same order of magnitude as the rounding error
            if rmse > inherent_rmse * 10:
                print(f"WARNING: Interpolation gradients are not close enough for {name} with interpolation atol {atol}; observed gradient RMSE is {rmse}")
            else:
                print(f"Interpolation gradients are close enough for {name} with atol {atol}; observed gradient RMSE is {rmse}")
            
