# Fitting the NODDI model

With this tutorial you can learn how to fit the [NODDI model](http://www.ncbi.nlm.nih.gov/pubmed/22484410) to a sample dataset.

## 1. Prepare dataset

We will use the sample dataset available from the [NODDI official website](http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab).

After unzipping, create the directory structure as follows:

```
└── Study01
    └── Subject01
        ├── NODDI_DWI.hdr
        ├── NODDI_DWI.img
        ├── NODDI_protocol.bval
        ├── NODDI_protocol.bvec
        ├── brain_mask.hdr
        ├── brain_mask.img
        ├── roi_mask.hdr
        └── roi_mask.img
```

## 2. DWI preprocess

Usually DWI images need some preprocessing (e.g., eddy current correction, head movement correction, and skull stripping).
You need to perform these pre-preprocessing steps before fitting the model.

Assuming this pre-processing has already been done for this sample dataset, we skip this step here.

## 3. Fitting the model to the data

Now, you are in the directory where `Study01` exists.

```bash
$ ls
Study01
```

Run the Python interpreter (of course, ipython is your best friend),

```bash
$ python
```

In the Python shell, import the AMICO library and setup/initialize the framework:

```python
>>> import amico
>>> amico.core.setup()
```

This step will precompute all the necessary rotation matrices and store them in `~/.dipy`. Note that this setup/initialization step is necessary only once.


### Load the data
Now, you can tell AMICO the location/directory containing all the data for this study/subject:

```python
>>> ae = amico.Evaluation("Study01", "Subject01")
```

If you don't have a scheme file, you can generate it from the bvals/bvecs files as follows:

```python
>>> amico.util.fsl2scheme("Study01/Subject01/NODDI_protocol.bval", "Study01/Subject01/NODDI_protocol.bvec")

-> Writing scheme file to [ Study01/Subject01/NODDI_protocol.scheme ]
'Study01/Subject01/NODDI_protocol.scheme'
```

Load the data:

```python
>>> ae.load_data(dwi_filename = "NODDI_DWI.img", scheme_filename = "NODDI_protocol.scheme", mask_filename = "roi_mask.img", b0_thr = 0)

-> Loading data:
	* DWI signal...
		- dim    = 128 x 128 x 50 x 81
		- pixdim = 1.875 x 1.875 x 2.500
	* Acquisition scheme...
		- 81 samples, 2 shells
		- 9 @ b=0 , 24 @ b=700.0 , 48 @ b=2000.0
	* Binary mask...
		- dim    = 128 x 128 x 50
		- pixdim = 1.875 x 1.875 x 2.500
		- voxels = 5478
   [ 0.2 seconds ]
```

### Compute the response functions
Set model for NODDI and generate the response functions for all the compartments:

```python
>>> ae.set_model("NODDI")
>>> ae.generate_kernels()

-> Creating LUT for "NODDI" model:
   [ 149.3 seconds ]
```

Note that you need to compute the reponse functions only once per study; in fact, scheme files with same b-values but different number/distribution of samples on each shell will result in the same precomputed kernels (which are actually computed at higher angular resolution). The function `generate_kernels()` does not recompute the kernels if they already exist, unless the flag `regenerate` is set, e.g. `generate_kernels( regenerate = True )`.

Load the precomputed kernels (at higher resolution) and adapt them to the actual scheme (distribution of points on each shell) of the current subject:

```python
>>> ae.load_kernels()

-> Resampling LUT for subject "Subject01":
   [ 52.2 seconds ]
```

### Model fit
It takes a little time depending on the number of voxels (but much much faster than the original NODDI).

```python
>>> ae.fit()

-> Fitting "NODDI" model to 5478 voxels:
   [ 00h 00m 09s ]
```

Finally, save the results as NIfTI images:

```python
>>> ae.save_results()

-> Saving output to "AMICO/NODDI/*":
	- configuration  [OK]
	- FIT_dir.nii.gz  [OK]
	- FIT_ICVF.nii.gz  [OK]
	- FIT_OD.nii.gz  [OK]
	- FIT_ISOVF.nii.gz  [OK]
   [ DONE ]
```

Well done!!

## 4. View results

You will find the estimated parameters of the NODDI model, i.e. ODI, ICVF and ISOVF, in the subject's subdirectory `AMICO/NODDI`.


```bash
$ ls Study01/Subject01/AMICO/NODDI/
FIT_ICVF.nii.gz		FIT_OD.nii.gz		config.pickle
FIT_ISOVF.nii.gz	FIT_dir.nii.gz
```

Open them with your favorite viewer. Congratulations!!

