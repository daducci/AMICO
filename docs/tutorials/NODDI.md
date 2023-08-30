# NODDI - Neurite Orientation Dispersion and Density Imaging
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neuroimage.2012.03.072-%23FAB70C?labelColor=%23363D45)](https://doi.org/10.1016/j.neuroimage.2012.03.072)

With this tutorial you will learn how to fit the `NODDI` model to a sample dataset.

## Prepare the dataset
Download the sample dataset from the NODDI official website:

- [NODDI_example_dataset.zip](https://www.nitrc.org/frs/download.php/11758/NODDI_example_dataset.zip)

After unzipping, your directory structure should look like this:
```Shell
NODDI_example_dataset/
├── brain_mask.hdr
├── brain_mask.img
├── NODDI_DWI.hdr
├── NODDI_DWI.img
├── NODDI_protocol.bval
└── NODDI_protocol.bvec
```

## Preprocess the data
Usually, DWI images need some preprocessing (e.g., eddy current correction, head movement correction, and skull stripping). You need to perform these pre-preprocessing steps before fitting the model. Assuming this pre-processing has already been done for this sample dataset, we skip those steps here.

## Run `AMICO`
### Initialization
Move into the `NODDI_example_dataset` directory and run a Python interpreter:
```Shell
$ cd NODDI_example_dataset
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

You can generate the scheme file from the bvals/bvecs files of your acquisition with the `fsl2scheme()` method:
```Python
>>> amico.util.fsl2scheme('NODDI_protocol.bval', 'NODDI_protocol.bvec')

-> Writing scheme file to [ NODDI_protocol.scheme ]
'NODDI_protocol.scheme'
```

### Load the data
Load your data with the `load_data()` method:
```Python
>>> ae.load_data('NODDI_DWI.img', 'NODDI_protocol.scheme', mask_filename='brain_mask.img', b0_thr=0)

-> Loading data:
	* DWI signal
		- dim    = 128 x 128 x 50 x 81
		- pixdim = 1.875 x 1.875 x 2.500
	* Acquisition scheme
		- 81 samples, 2 shells
		- 9 @ b=0 , 24 @ b=700.0 , 48 @ b=2000.0 
	* Binary mask
		- dim    = 128 x 128 x 50
		- pixdim = 1.875 x 1.875 x 2.500
		- voxels = 178924
   [ 0.3 seconds ]

-> Preprocessing:
	* Normalizing to b0... [ min=0.00,  mean=2.78, max=2862.00 ]
	* Keeping all b0 volume(s)
   [ 8.2 seconds ]
```

### Compute the response functions
Set the `NODDI` model with the `set_model()` method and generate the response functions with the `generate_kernels()` method:
```Python
>>> ae.set_model('NODDI')
>>> ae.generate_kernels(regenerate=True)

-> Creating LUT for "NODDI" model:
   [ 1.8 seconds ]
```
!!! note
	You need to compute the reponse functions only once per study; in fact, scheme files with same b-values but different number/distribution of samples on each shell will result in the same precomputed kernels (which are actually computed at higher angular resolution). The method `generate_kernels()` does not recompute the kernels if they already exist, unless the flag `regenerate` is set, e.g. `generate_kernels(regenerate=True)`.

Load the precomputed kernels (at higher resolution) and adapt them to the actual scheme (distribution of points on each shell) of the current subject with the `load_kernels()` method:
```Python
>>> ae.load_kernels()

-> Resampling LUT for subject ".":
   [ 0.4 seconds ]
```

### Fit the model to the data
Fit the model to the data with the `fit()` method:
```Python
>>> ae.fit()

-> Estimating principal directions (OLS):
   [ 00h 00m 01s ]

-> Fitting 'NODDI' model to 178924 voxels (using 32 threads):
   [ 00h 00m 04s ]
```

### Save the results
Finally, save the results as NIfTI images with the `save_results()` method:
```Python
>>> ae.save_results()

-> Saving output to "AMICO/NODDI/*":
	- configuration  [OK]
	- fit_dir.nii.gz  [OK]
	- fit_NDI.nii.gz  [OK]
	- fit_ODI.nii.gz  [OK]
	- fit_FWF.nii.gz  [OK]
   [ DONE ]
```

## Visualize the results
🎉Congratulations! You have successfully fitted the `NODDI` model to your data.🎉 You will find the estimated parameters in the `NODDI_example_dataset/AMICO/NODDI` directory:
```Shell
NODDI_example_dataset/AMICO/NODDI/
├── config.pickle
├── fit_dir.nii.gz
├── fit_FWF.nii.gz
├── fit_NDI.nii.gz
└── fit_ODI.nii.gz
```
Open them with your favorite viewer.
