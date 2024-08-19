# TODO: Currently exceeds error thresholds, need to debug why
import multiprocessing as mp
import numpy as np # Just utility
import opensimplex # For test functions
import SamplingMesh

def parabola(x):
    return np.sum(x**2)

def simplex_2d(x):
    return opensimplex.noise2(*x)
    
def simplex_3d(x):
    return opensimplex.noise3(*x)

def simplex_4d(x):
    return opensimplex.noise4(*x)

if __name__ == '__main__':
    tests = [("1D parabola", 1, parabola),
             ("2D noise", 2, simplex_2d),
             ("3D noise", 3, simplex_3d),
             ("4D noise", 4, simplex_4d)]
    
    for name, ndim, f in tests:
        SamplingMesh.NDIM = ndim
        for atol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
            with mp.Pool() as pool:
                mesh = SamplingMesh.SamplingMesh(f, pool, atol=atol, rtol=0)
                xs = np.random.normal(size = (100, SamplingMesh.NDIM))
                try:
                    ys = mesh.multi_interpolate(xs)
                except Exception as e:
                    print(f"Encountered error on {name} with atol {atol}")
                    raise e
            
            rmse = np.sqrt(np.mean((ys - np.apply_along_axis(f, -1, xs))**2))
            if rmse  > mesh.atol:
                print(f"Error too high: RMSE of {rmse} exceeds target of {mesh.atol} for {name}.")
