# sampling-mesh
An adaptively scaled interpolation mesh for accelerating repeated function calls in Python.

This has similar goals as a caching function wrapper, but for continuously valued inputs.

This is a work in progress.

Check out SamplingMesh.py for the centerpiece of the project, the SamplingMesh class. This class allows a user to specify an (absolute and/or relative) error tolerance, a function, and a resolution for initial curvature estimates. From this, it creates a mesh (structured as a k-d tree) that allows repeated function calls to the same region to reuse old calculations. This is accomplished by finding the cell the requested point is in, halving the size of the cell in all directions until the root mean squared error (RMSE) is within the requested bounds.

Due to fleshing out a full hypercube cell, the first interpolation call to the mesh calls the target function $\mathcal{O} \left(2^{k}\right)$ times, where $k$ is the number of dimensions. However, subsequent calls to that cell don't call the target function at all. If you expect to be calling the function many times over the same region, this can save significant time.

Incoming features: gradient calculation, adaptive cell ratio determination.
