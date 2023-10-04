# SANDI - Soma And Neurite Density Imaging
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neuroimage.2020.116835-%23FAB70C?labelColor=%23363D45)](https://doi.org/10.1016/j.neuroimage.2020.116835)

With this tutorial you will learn how to fit the `SANDI` model to a sample dataset.
!!! important
	- Default setting we present here should be used only on data with at least 5 b shells. Although we strongly discourage it, to fit the model to data with less b shells, one or more model parameters must be fixed (e.g., for data with only 3 shells, 2 of the 5 model parameters must be fixed to a reasonable value).
	- Interpretation of the compartments of this model as meaningful biological structures holds under specific assumptions that can be valid in specific experimental regimes (see [Palombo, Marco, et al. 2020](https://doi.org/10.1016/j.neuroimage.2020.116835)). In particular, for in vivo studies, diffusion time <~25ams and some data at b>3000$s/mm^2$. Although we strongly discourage it, the model can be used to analyze also data outside this experimental regime, but then the biological interpretation of the model parameters should be taken vary carefully.

## Prepare the dataset
Download the sample dataset available from the BrainHack repository:

- [BrainHack repository](https://drive.google.com/drive/folders/1FJ-mg-UOmH9HFnBYWrlqJ00vT1FMHwY2?usp=sharing)
!!! note
	These data were kindly provided by Dr. Andrada Ianus and Dr. Noam Shemesh were acquired in the Preclinical MRI Lab at Champalimaud Foundation.

After unzipping, your directory structure should look like this:
```bash
SANDI_Data_BrainHack_Shared/
├── bvals.bval
├── bvecs.bvec
├── MouseData_Slice_Mask.nii
└── MouseData_Slice.nii
```

## Preprocess the data
Usually, DWI images need some preprocessing (e.g., eddy current correction, head movement correction, and skull stripping). You need to perform these pre-preprocessing steps before fitting the model. Assuming this pre-processing has already been done for this sample dataset, we skip those steps here.

## Run `AMICO`
### Initialization
Move into the `SANDI_Data_BrainHack_Shared` directory and run a Python interpreter:
```bash
$ cd SANDI_Data_BrainHack_Shared
$ python
```

In the Python shell, import the `AMICO` library and setup/initialize the framework:
```python
>>> import amico
>>> amico.setup()

-> Precomputing rotation matrices:
   [ DONE ]
```
!!! note
	This step will precompute all the necessary rotation matrices and store them in `~/.dipy`. This initialization step is necessary only the first time you use `AMICO`.

Now you can instantiate an `Evaluation` object and start the analysis:
```python
>>> ae = amico.Evaluation()
```

To fit the `SANDI` model you also need to perform a directional average on your data. You can do it by setting the `doDirectionalAverage` flag to `True` with the `set_config()` method:
```python
>>> ae.set_config('doDirectionalAverage', True)
```

You can generate the scheme file from the bvals/bvecs files and values of delta and small_delta of your acquisition with the `sandi2scheme()` method:
```python
>>> delta = 0.02 # Time between pulses [s]
>>> small_delta = 0.0055 # Pulses duration in [s]
>>> TE = 0.03 # Echo time if different from delta+small_delta [s] (optional)
>>> amico.util.sandi2scheme('bvals.bval', 'bvecs.bvec', delta, small_delta, TE_data=TE, schemeFilename='SANDI_scheme.txt', bStep=100)

-> Rounding b-values to nearest multiple of 100.0

-> Writing scheme file to [ SANDI_scheme.txt ]
'SANDI_scheme.txt'
```
!!! note
	TE is not mandatory and if not provided it will be computed internally as the sum of delta and small_delta. Moreover, in case delta and small_delta (and echo time) have different values for different directions and/or shell the corresponding files can be provided in the same format as bvals.bval.

### Load the data
Load your data with the `load_data()` method:
```python
>>> ae.load_data(dwi_filename='MouseData_Slice.nii', scheme_filename='SANDI_scheme.txt', mask_filename='MouseData_Slice_Mask.nii', b0_thr=10)

-> Loading data:
	* DWI signal
		- dim    = 118 x 100 x 2 x 336
		- pixdim = 0.120 x 0.120 x 0.400
	* Acquisition scheme
		- 336 samples, 8 shells
		- 16 @ b=0 , 40 @ b=1000.0 , 40 @ b=4000.0 , 40 @ b=7000.0 , 40 @ b=10000.0 , 40 @ b=2500.0 , 40 @ b=5500.0 , 40 @ b=8500.0 , 40 @ b=12500.0 
	* Binary mask
		- dim    = 118 x 100 x 2
		- pixdim = 0.120 x 0.120 x 0.400
		- voxels = 8107
   [ 0.1 seconds ]

-> Preprocessing:
	* Normalizing to b0... [ min=0.00,  mean=0.47, max=3.96 ]
	* Keeping all b0 volume(s)
	* Performing the directional average on the signal of each shell... 
		- dim    = 118 x 100 x 2 x 9
		- pixdim = 0.120 x 0.120 x 0.400
	* Acquisition scheme
		- 9 samples, 8 shells
		- 1 @ b=0 , 1 @ b=1000.0 , 1 @ b=2500.0 , 1 @ b=4000.0 , 1 @ b=5500.0 , 1 @ b=7000.0 , 1 @ b=8500.0 , 1 @ b=10000.0 , 1 @ b=12500.0 
   [ 0.1 seconds ]
```

### Compute the response functions
Set the `SANDI` model with the `set_model()` method and generate the response functions with the `generate_kernels()` method:
```python
>>> ae.set_model('SANDI')
>>> d_is = 3e-3 # Intra-soma diffusivity [mm^2/s]
>>> Rs = np.array([1.55555556e-06, 3.44444444e-06, 4.44444444e-06, 5.33333333e-06, 6.00000000e-06, 6.55555556e-06, 8.11111111e-06, 9.55555556e-06, 1.16666667e-05]) # Radii of the soma [meters]
>>> d_in = np.array([0.00091667, 0.00169444, 0.003]) # Intra-neurite diffusivitie(s) [mm^2/s]
>>> d_isos = np.array([0.00036111, 0.00163889, 0.003]) # Extra-cellular isotropic mean diffusivitie(s) [mm^2/s]
>>> ae.model.set(d_is, Rs, d_in, d_isos)
>>> ae.generate_kernels(regenerate=True, ndirs=1)

-> Creating LUT for "SANDI" model:
   [ 0.8 seconds ]
```
!!! note
	- The dictionary provided as default inside the code was optimized for the specific acquisition of the data in SANDI_Data_BrainHack_Shared. If you need to change them you can modify model-specific variables and then set the model with the `set()` method.
	- You need to compute the reponse functions only once per study; in fact, scheme files with same b-values but different number/distribution of samples on each shell will result in the same precomputed kernels (which are actually computed at higher angular resolution). The method `generate_kernels()` does not recompute the kernels if they already exist, unless the flag `regenerate` is set, e.g. `generate_kernels(regenerate=True)`.

Load the precomputed kernels (at higher resolution) and adapt them to the actual scheme (distribution of points on each shell) of the current subject with the `load_kernels()` method:
```python
>>> ae.load_kernels()

-> Resampling LUT for subject ".":
   [ 0.0 seconds ]
```

### Fit the model to the data
Fit the model to the data with the `fit()` method:
```python
>>> lambda1 = 0 # L1 regularization term (MUST be varied according to data)
>>> lambda2 = 5e-3 # L2 regularization term (MUST be varied according to data)
>>> ae.set_solver(lambda1=lambda1, lambda2=lambda2)
>>> ae.fit()

-> Fitting 'SANDI' model to 8107 voxels (using 32 threads):
   [ 00h 00m 00s ]
```
!!! note
	Now you need to set the regularisation parameters according to your data and if you need to apply L1 or L2 regularization or a combination of both.

### Save the results
Finally, save the results as NIfTI images with the `save_results()` method:
```python
>>> ae.save_results(save_dir_avg=True)

-> Saving output to "AMICO/SANDI/*":
	- configuration  [OK]
	- fit_fsoma.nii.gz  [OK]
	- fit_fneurite.nii.gz  [OK]
	- fit_fextra.nii.gz  [OK]
	- fit_Rsoma.nii.gz  [OK]
	- fit_Din.nii.gz  [OK]
	- fit_De.nii.gz  [OK]
	- dir_avg_signal.nii.gz  [OK]
	- dir_avg.scheme  [OK]
   [ DONE ]
```

## Visualize the results
🎉Congratulations! You have successfully fitted the `SANDI` model to your data.🎉 You will find the estimated parameters in the `SANDI_Data_BrainHack_Shared/AMICO/SANDI` directory:
```bash
SANDI_Data_BrainHack_Shared/AMICO/SANDI/
├── config.pickle
├── dir_avg.scheme
├── dir_avg_signal.nii.gz
├── fit_De.nii.gz
├── fit_Din.nii.gz
├── fit_fextra.nii.gz
├── fit_fneurite.nii.gz
├── fit_fsoma.nii.gz
└── fit_Rsoma.nii.gz
```
Open them with your favorite viewer.
