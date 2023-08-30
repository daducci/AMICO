# ActiveAx
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neuroimage.2010.05.043-%23FAB70C?labelColor=%23363D45)](https://doi.org/10.1016/j.neuroimage.2010.05.043)

With this tutorial you will learn how to fit the `ActiveAx` model to a sample dataset.

## Prepare the dataset
Download the sample DWI data:

- [DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b13183_3b0_N90.nii.gz](https://osf.io/download/udc7v/)
- [DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b1925_3b0_N90.nii.gz](https://osf.io/download/he4aj/)
- [DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b1931_3b0_N90.nii.gz](https://osf.io/download/9avhm/)
- [DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b3091_3b0_N90.nii.gz](https://osf.io/download/fkpm3/)

Merge the downloaded datasets into one:
```Shell
$ export FSLOUTPUTTYPE=NIFTI_GZ
$ fslmerge -t DWI.nii.gz DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b13183_3b0_N90.nii.gz DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b1925_3b0_N90.nii.gz DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b1931_3b0_N90.nii.gz DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b3091_3b0_N90.nii.gz
```

Download the scheme file and the binary mask of the corpus callosum:

- [ActiveAxG140_PM.scheme1](http://web4.cs.ucl.ac.uk/research/medic/camino/pmwiki/uploads/Tutorials/ActiveAxG140_PM.scheme1)
- [ActiveAx_Tutorial_MidSagCC.nii](http://hardi.epfl.ch/static/data/AMICO_demos/ActiveAx_Tutorial_MidSagCC.nii)

Move all the files into a directory named `sub_01`. Your directory structure should look like this:
```Shell
sub_01/
├── ActiveAxG140_PM.scheme1
├── ActiveAx_Tutorial_MidSagCC.nii
├── DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b13183_3b0_N90.nii.gz
├── DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b1925_3b0_N90.nii.gz
├── DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b1931_3b0_N90.nii.gz
├── DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b3091_3b0_N90.nii.gz
└── DWI.nii.gz
```

## Preprocess the data
Usually, DWI images need some preprocessing (e.g., eddy current correction, head movement correction, and skull stripping). You need to perform these pre-preprocessing steps before fitting the model. Assuming this pre-processing has already been done for this sample dataset, we skip those steps here.

## Run `AMICO`
### Initialization
Move into the `sub_01` directory and run a Python interpreter:
```Shell
$ cd sub_01
$ python
```

In the Python shell, import the `AMICO` library and setup/initialize the framework:
```Python
>>> import amico
>>> amico.setup()

-> Precomputing rotation matrices:
   [ DONE ]
```
!!! note
	This step will precompute all the necessary rotation matrices and store them in `~/.dipy`. This initialization step is necessary only the first time you use `AMICO`.

Now you can instantiate an `Evaluation` object and start the analysis:
```Python
>>> ae = amico.Evaluation()
```

### Load the data
Load your data with the `load_data()` method:
```Python
>>> ae.load_data(dwi_filename='DWI.nii.gz', scheme_filename='ActiveAxG140_PM.scheme1', mask_filename='ActiveAx_Tutorial_MidSagCC.nii', b0_thr=0)

-> Loading data:
	* DWI signal
		- dim    = 128 x 256 x 3 x 372
		- pixdim = 0.400 x 0.400 x 0.500
	* Acquisition scheme
		- 372 samples, 4 shells
		- 12 @ b=0 , 90 @ b=13199.8 , 90 @ b=1926.6 , 90 @ b=1933.3 , 90 @ b=3095.8 
	* Binary mask
		- dim    = 128 x 256 x 3
		- pixdim = 0.400 x 0.400 x 0.500
		- voxels = 338
   [ 0.6 seconds ]

-> Preprocessing:
	* Normalizing to b0... [ min=0.00,  mean=0.79, max=8.46 ]
	* Keeping all b0 volume(s)
   [ 0.2 seconds ]
```

### Compute the response functions
Set the `CylinderZeppelinBall` model with the `set_model()` method and generate the response functions with the `generate_kernels()` method:
```Python
>>> ae.set_model('CylinderZeppelinBall')
>>> ae.generate_kernels(regenerate=True)

-> Creating LUT for "Cylinder-Zeppelin-Ball" model:
   [ 1.7 seconds ]
```
!!! note
	You need to compute the reponse functions only once per study; in fact, scheme files with same b-values but different number/distribution of samples on each shell will result in the same precomputed kernels (which are actually computed at higher angular resolution). The method `generate_kernels()` does not recompute the kernels if they already exist, unless the flag `regenerate` is set, e.g. `generate_kernels(regenerate=True)`.

Load the precomputed kernels (at higher resolution) and adapt them to the actual scheme (distribution of points on each shell) of the current subject with the `load_kernels()` method:
```Python
>>> ae.load_kernels()

-> Resampling LUT for subject ".":
   [ 0.2 seconds ]
```

### Fit the model to the data
Fit the model to the data with the `fit()` method:
```Python
>>> ae.fit()

-> Estimating principal directions (OLS):
   [ 00h 00m 00s ]

-> Fitting 'Cylinder-Zeppelin-Ball' model to 338 voxels (using 32 threads):
   [ 00h 00m 00s ]
```

### Save the results
Finally, save the results as NIfTI images with the `save_results()` method:
```Python
>>> ae.save_results()

-> Saving output to "AMICO/CylinderZeppelinBall/*":
	- configuration  [OK]
	- fit_dir.nii.gz  [OK]
	- fit_v.nii.gz  [OK]
	- fit_a.nii.gz  [OK]
	- fit_d.nii.gz  [OK]
   [ DONE ]
```

## Visualize the results
🎉Congratulations! You have successfully fitted the `ActiveAx` model to your data.🎉 You will find the estimated parameters in the `sub_01/AMICO/CylinderZeppelinBall` directory:
```Shell
sub_01/AMICO/CylinderZeppelinBall/
├── config.pickle
├── fit_a.nii.gz
├── fit_dir.nii.gz
├── fit_d.nii.gz
└── fit_v.nii.gz
```
Open them with your favorite viewer.
