import multiprocessing as mp
import numpy as np # Just utility
import SamplingMesh

def parabola(x):
    return np.sum(x**2)

if __name__ == '__main__':
    #SamplingMesh.DEBUG = True
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
