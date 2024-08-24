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

def complex_square(x):
    return np.array([x[0]**2 - x[1]**2, 2 * x[0] * x[1]])


if __name__ == '__main__':
    #SamplingMesh.DEBUG = True
    tests = [("1D parabola", 1, None, parabola, np.ones(1)),
             ("2D noise", 2, None, simplex_2d, np.ones(2) * 0.5),
             ("3D noise", 3, None, simplex_3d, np.ones(3) * 0.5),
             ("4D noise", 4, None, simplex_4d, np.ones(4) * 0.5),
             ("complex squaring", 2, 2, complex_square, np.ones(2))]
    
    print("Testing atol")
    
    for name, ndim, outdim, f, resolution in tests:
        for atol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
            with mp.Pool() as pool:
                mesh = SamplingMesh.SamplingMesh(f, pool, ndim, outdim, scale=resolution, atol=atol, rtol=0)
                xs = np.random.normal(size = (100, ndim))
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
    
    print("Testing rtol")
    
    for name, ndim, outdim, f, resolution in tests:
        for rtol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
            with mp.Pool() as pool:
                mesh = SamplingMesh.SamplingMesh(f, pool, ndim, outdim, scale=resolution, atol=1e-10, rtol=rtol)
                xs = np.random.normal(size = (100, ndim))
                try:
                    ys = mesh.multi_interpolate(xs)
                except Exception as e:
                    print(f"Encountered error on {name} with rtol {rtol}")
                    raise e
            
            correct = np.apply_along_axis(f, -1, xs)
            rmse = np.sqrt(np.mean((ys - correct)**2))
            cutoff = mesh.atol + mesh.rtol * np.mean(np.abs(correct))
            if rmse  > cutoff:
                print(f"WARNING: RMSE of {rmse} exceeds target of {cutoff} for {name}.")
            else:
                print(f"RMSE of {rmse} within {cutoff} for {name}.")
