# What's new
### New features and improvements in `AMICO` v2.0.0
## Multithreading
`AMICO` now uses multithreading insted of multiprocessing. This allows to further speed up the fitting process. See the [multithreading](configuration/multithreading.md) section for more information.

## Principal diffusion direction
The principal diffusion direction is now computed by default using the Ordinary Least Squares (OLS) method instead of the Weighted Least Squares (WLS) method. You can still use the WLS method by setting the `DTI_fit_method` parameter to `WLS`. See the [configuration parameters](configuration/config_params.md) section for more information.

## Configuration parameters
__`study_path` and `subject`__<br>
Default `study_path` and `subject` parameters in the `Evaluation` class are now set to `'.'` instead of `None`. This means that if you don't set these parameters, `AMICO` will look for the data in the current directory.

__`ndirs`__<br>
Default `ndirs` parameter in the `generate_kernels()` method is now set to `500` instead of `32761`.

__`DTI_fit_method`__<br>
New parameter to set the method used to fit the diffusion tensor model to compute the principal diffusion directions.

__`n_threads`__<br>
Now the parameter to set the number of threads used by `AMICO` during the model fitting is `n_threads` instead of `parallel_jobs`.

__`BLAS_n_threads`__<br>
New parameter to set the number of threads used by BLAS libraries.

__`doComputeRMSE`__<br>
New parameter to compute the Root Mean Square Error (RMSE) between the predicted and the measured signal.

__`doSaveModulatedMaps`__<br>
New parameter (specific for the `NODDI` model) to compute the modulated `NDI` and `ODI` maps for [Tissue-weighted mean](https://csparker.github.io/research/2021/11/16/Tissue-weighted-mean.html) analysis.

## NODDI maps names
The NODDI maps are called with names in accordance with the original paper:
- `NDI` instead of `ICVF`
- `ODI` instead of `OD`
- `FWF` instead of `FISO`


## License
`AMICO` is now released under a proprietary license which allows free use for non-commercial purposes. If you are interested in using `AMICO` for commercial purposes, please contact _Alessandro Daducci_ at alessandro.daducci@univr.it. You can read the full license text [here](https://github.com/daducci/AMICO/blob/master/LICENSE).
