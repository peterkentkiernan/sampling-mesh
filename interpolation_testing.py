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
    #SamplingMesh.DEBUG = True
    tests = [("1D parabola", 1, parabola, np.ones(1)),
             ("2D parabola", 2, parabola, np.ones(2)),
             ("3D parabola", 3, parabola, np.ones(3)),
             ("2D noise", 2, simplex_2d, np.ones(2) * 0.5),
             ("3D noise", 3, simplex_3d, np.ones(3) * 0.5),
             ("4D noise", 4, simplex_4d, np.ones(4) * 0.5)]
    
    for name, ndim, f, resolution in tests:
        SamplingMesh.NDIM = ndim
        for atol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
            with mp.Pool() as pool:
                mesh = SamplingMesh.SamplingMesh(f, pool, scale=resolution, atol=atol, rtol=0)
                xs = np.random.normal(size = (100, SamplingMesh.NDIM))
                try:
                    ys = mesh.multi_interpolate(xs)
                except Exception as e:
                    print(f"Encountered error on {name} with atol {atol}")
                    raise e
            
            rmse = np.sqrt(np.mean((ys - np.apply_along_axis(f, -1, xs))**2))
            if rmse  > mesh.atol:
                print(f"WARNING: RMSE of {rmse} exceeds target of {mesh.atol} for {name}.")
            else:
                print(f"RMSE of {rmse} within {mesh.atol} for {name}.")
