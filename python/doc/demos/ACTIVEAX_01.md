# Fitting the ACTIVEAX model

With this tutorial you can learn how to fit the [ActiveAx model](http://www.ncbi.nlm.nih.gov/pubmed/20580932) to a sample dataset.

## 1. Prepare dataset

1. Download the original DWI data from [here](http://dig.drcmr.dk/activeax-dataset/) and save it into a folder named Study02/Subject01.

2. Merge the downloaded datasets into one:

    ```bash
    export FSLOUTPUTTYPE=NIFTI
    fslmerge -t Study02/Subject01/DWI.nii \\
    Study02/Subject01/DRCMR_ActiveAx4CCfit_E2503_Mbrain1_B13183_3B0_ELEC_N90_Scan1_DIG.nii \\
    Study02/Subject01/DRCMR_ActiveAx4CCfit_E2503_Mbrain1_B1925_3B0_ELEC_N90_Scan1_DIG.nii \\
    Study02/Subject01/DRCMR_ActiveAx4CCfit_E2503_Mbrain1_B1931_3B0_ELEC_N90_Scan1_DIG.nii \\
    Study02/Subject01/DRCMR_ActiveAx4CCfit_E2503_Mbrain1_B3091_3B0_ELEC_N90_Scan1_DIG.nii
    ```

3. Download the scheme file from [here](http://cmic.cs.ucl.ac.uk/camino/uploads/Tutorials/ActiveAxG140_PM.scheme1) and save it into the same folder.

4. Download the binary mask of the corpus callosum from [here](http://hardi.epfl.ch/static/data/AMICO_demos/ActiveAx_Tutorial_MidSagCC.nii) to the same folder.

5. You should have the following folder structure:

```
└── Study02
    └── Subject01
        ├── DRCMR_ActiveAx4CCfit_E2503_Mbrain1_B13183_3B0_ELEC_N90_Scan1_DIG.nii
        ├── DRCMR_ActiveAx4CCfit_E2503_Mbrain1_B1925_3B0_ELEC_N90_Scan1_DIG.nii
        ├── DRCMR_ActiveAx4CCfit_E2503_Mbrain1_B1931_3B0_ELEC_N90_Scan1_DIG.nii
        ├── DRCMR_ActiveAx4CCfit_E2503_Mbrain1_B3091_3B0_ELEC_N90_Scan1_DIG.nii
        ├── DWI.nii
        ├── ActiveAxG140_PM.scheme1
        └── ActiveAx_Tutorial_MidSagCC.nii
```

## 2. Fitting the model to the data

Now, you are in the directory where `Study02` exists.

```bash
$ ls
Study02
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
>>> ae = amico.Evaluation("Study02", "Subject01")
```

Load the data:

```python
>>> ae.load_data(dwi_filename = "DWI.nii", scheme_filename = "ActiveAxG140_PM.scheme1", mask_filename = "ActiveAx_Tutorial_MidSagCC.nii", b0_thr = 0)

—> Loading data:
	* DWI signal...
		- dim    = 128 x 256 x 3 x 372
		- pixdim = 0.400 x 0.400 x 0.500
	* Acquisition scheme...
		- 372 samples, 4 shells
		- 12 @ b=0 , 90 @ b=13191.3 , 90 @ b=1925.4 , 90 @ b=1932.0 , 90 @ b=3093.8
	* Binary mask...
		- dim    = 128 x 256 x 3
		- pixdim = 0.400 x 0.400 x 0.500
		- voxels = 338
   [ 0.1 seconds ]
```

### Compute the response functions
Set model for ACTIVEAX and generate the response functions for all the compartments. The ACTIVEAX model is a three compartment model that combines the Cylinder, Zeppelin and Ball models:

```python
>>> ae.set_model("CylinderZeppelinBall")
>>> ae.generate_kernels()

—> Creating LUT for "Cylinder-Zeppelin-Ball" model:
   [ 106.0 seconds ]
```

Note that you need to compute the reponse functions only once per study; in fact, scheme files with same b-values but different number/distribution of samples on each shell will result in the same precomputed kernels (which are actually computed at higher angular resolution). The function `generate_kernels()` does not recompute the kernels if they already exist, unless the flag `regenerate` is set, e.g. `generate_kernels( regenerate = True )`.

Load the precomputed kernels (at higher resolution) and adapt them to the actual scheme (distribution of points on each shell) of the current subject:

```python
>>> ae.load_kernels()

—> Resampling LUT for subject "Subject01":
   [ 92.4 seconds ]
```

### Model fit
It takes a little time depending on the number of voxels (but much much faster than the original ACTIVEAX).

```python
>>> ae.fit()

—> Fitting "Cylinder-Zeppelin-Ball" model to 338 voxels:
   [ 6.2 seconds ]
```

Finally, save the results as NIfTI images:

```python
>>> ae.save_results()

—> Saving output to "AMICO/CylinderZeppelinBall/*":
	- configuration  [OK]
	- FIT_dir.nii.gz  [OK]
	- FIT_v.nii.gz  [OK]
	- FIT_a.nii.gz  [OK]
	- FIT_d.nii.gz  [OK]
   [ DONE ]
```

Well done!!

## 4. View results

You will find the estimated parameters of the ACTIVEAX model, i.e. Axon diameter index, ICVF and axon density, in the subject's subdirectory `AMICO/CylinderZeppelinBall`.


```bash
$ ls Study02/Subject01/AMICO/CylinderZeppelinBall/
FIT_a.nii.gz	FIT_d.nii.gz		config.pickle
FIT_v.nii.gz	FIT_dir.nii.gz
```

Open them with your favorite viewer. Congratulations!!

