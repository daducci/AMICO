# Fitting the NODDI model

Here you can learn how to the NODDI model to a sample dataset.

## 1. Prepare dataset

The sample dataset is available from [NODDI official website](http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab).

After unzip, make the directory structure as follows:

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
You need do these preprocessing before fitting the model.

Assuming these preprocessing has been already done for this sample dataset, we skip this step here.

## 3. Fitting the NODDI model

Now, you are in the directory where `Study01` exists.

```bash
$ ls
Study01
```

Run the python interpreter (of course, ipython is your best friend),

```bash
$ python
```

Then, in python, you can fit the NODDI model. First, import AMICO:

```python
>>> import amico
```

If you don't have a scheme file, generate it from bval/bvec files:

```python
>>> amico.util.fsl2scheme("Study01/Subject01/NODDI_protocol.bval", "Study01/Subject01/NODDI_protocol.bvec")

-> Writing scheme file to [ Study01/Subject01/NODDI_protocol.scheme ]
'Study01/Subject01/NODDI_protocol.scheme'
```

Next, setup/initialize the AMICO framework. This generates a precomputed rotation matrix in `~/.dypy`.
Note that this setup/initialize is necessary only once.

```python
>>> amico.core.setup()
```

Tell the study and subject directory:

```python
>>> ae = amico.Evaluation("Study01", "Subject01")
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

Set model for NODDI:

```python
>>> ae.set_model("NODDI")
```

Generate kernels. You need this only once per study (i.e., the same scheme files result in the same kernels).
Note that `generate_kernels()` do nothing if the kernels already exists.

```python
>>> ae.generate_kernels(lmax = 12)

-> Creating LUT for "NODDI" model:
   [ 149.3 seconds ]
```

Load the kernels:

```python
>>> ae.load_kernels()

-> Resampling LUT for subject "Subject01":
   [ 52.2 seconds ]
```

Fit the model. It takes a little time depending on the number of voxels (but much much faster than the original NODDI).

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

