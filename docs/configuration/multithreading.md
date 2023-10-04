# Multithreading
## Fitting threads
By default, `AMICO` uses all the available cores on your machine to speed up the fitting process. However, you can specify the number of threads to use with the `n_threads` configuration parameter:
```python
>>> import amico

>>> ae = amico.Evaluation()
>>> ae.set_config('n_threads', 4)
```
!!! important
    The `n_threads` parameter can be set to any positive integer value. If you set it to `-1`, `AMICO` will use all the available CPUs on your machine.

## BLAS threads
`AMICO` limits to `1` the number of threads used in the threadpool-backend of common BLAS implementations (e.g. `OpenBLAS`). This is done optimize light computations and avoid oversubscription of resources when using multithreaded methods of other packages (e.g. `numpy.linalg`). You can change this behaviour by setting the `BLAS_n_threads` configuration parameter:
```python
>>> import amico

>>> ae = amico.Evaluation()
>>> ae.set_config('BLAS_nthreads', 4)
```
!!! important
    The `BLAS_n_threads` parameter can be set to any positive integer value. If you set it to `-1`, `AMICO` will use all the available CPUs on your machine.
