# Configuration parameters
You can set the configuration parameters of `AMICO` by calling the `set_config()` method of the `Evaluation` class:
```python
>>> import amico

>>> ae = amico.Evaluation()
>>> ae.set_config('config_name', value)
```

## General parameters
__`peaks_filename`__<br>Peaks filename containing the main diffusion orientations. If `None`, the main diffusion orientations are computed using the diffusion tensor model of `dipy`. (default is `None`)

__`doNormalizeSignal`__<br>Normalize the signal to the b0 value. (default is `True`)

__`doKeepb0Intact`__<br>Keep the b0 image intact in the predicted signal. (default is `False`)

__`doComputeRMSE`__<br>Compute the root mean square error between the predicted and the measured signal. (default is `False`)

__`doComputeNRMSE`__<br>Compute the normalized root mean square error between the predicted and the measured signal. (default is `False`)

__`doMergeB0`__<br>Merge the b0 images into a single volume. (default is `False`)

__`doDebiasSignal`__<br>Remove Rician bias from the signal. See the [Rician bias](rician_bias.md) section for more information. (default is `False`)

__`DWI-SNR`__<br>Signal-to-noise ratio of the DWI data (SNR = b0/sigma). See the [Rician bias](rician_bias.md) section for more information. (default is `None`)
!!! important
    This parameter must be set if the `doDebiasSignal` parameter is set to `True`

## Model-specific parameters
### NODDI
__`doSaveModulatedMaps`__<br>Save the modulated NDI and ODI maps for tissue-weighted means described in [Parker, Christopher S. et al. 2021](https://doi.org/10.1016/j.neuroimage.2021.118749). (default is `False`)

### FreeWater
__`doSaveCorrectedDWI`__<br>Save the corrected DWI. (default is `False`)

### SANDI
__`doDirectionalAverage`__<br>Perform the directional average of the signal of each shell. (default is `False`)
!!! important
    This parameter must be set to `True` if you want to use the `SANDI` model

## Advanced parameters
!!! warning
    The following parameters should be used with caution. If you are not sure about what you are doing, please leave them to their default values

__`DTI_fit_method`__<br>Fit method for the diffusion tensor model. Possible values are `OLS` (ordinary least squares) and `WLS` (weighted least squares). (default is `OLS`)

__`n_threads`__<br>Number of threads to use during the model fitting. If `-1` the number of threads is set to the number of available CPUs in the system. See the [multithreading](multithreading.md) section for more information. (default is `-1`)

__`BLAS_n_threads`__<br>Number of threads to use in the threadpool-backend of common BLAS implementations. If `-1` the number of threads is set to the number of available CPUs in the system. See the [multithreading](multithreading.md) section for more information. (dafault is `1`)
