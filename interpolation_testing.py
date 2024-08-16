import multiprocessing as mp
import opensimplex
import SamplingMesh

tests = [("1D parabola", 1, lambda x: x[0]**2),
         ("2D noise", 2, lambda x: opensimplex.noise2(*x)),
         ("3D noise", 3, lambda x: opensimplex.noise3(*x)),
         ("4D noise", 4, lambda x: opensimplex.noise4(*x))]

for name, ndim, f in tests:
    SamplingMesh.NDIM = ndim
    for atol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        with mp.Pool() as pool:
            mesh = SamplingMesh.SamplingMesh(f, pool)
            mesh.rtol = 0
            mesh.atol = atol
            xs = np.random.normal(size = (100, NDIM))
            ys = mesh.multi_interpolate(xs)
        
        rmse = np.sqrt(np.mean((ys - np.apply_along_axis(f, -1, xs))**2))
        if rmse  > mesh.atol:
            raise RuntimeError(f"Error too high: RMSE of {rmse} exceeds target of {mesh.atol} for {name}.")

