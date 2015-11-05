# Tutorial for python-AMICO/NODDI

Here you can learn how to run python-AMICO/NODII for sampel dateset.

## 1. Prepare dataset

The sample dataset is available from [NODDI official website](http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab).

After unzip, make the directory strucutre as followings:

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

Usually you need some preprocesses for DWI images (e.g., eddy current correction, head movement correction, and skull stripping).
You need do these preprocesses before performing python-AMICO/NODDI.

Assuming these preprocesses have been done for the sample dataset, we skip this step here.

## 3. Performe python-AMICO/NODDI

Now, you are in the directory where `Study01` exists.

```bash
$ ls
Study01
```

Then, run python (of course, ipython is your best friend),

```bash
$ python
```

Then, in python, you can perform python-AMICO/NODDI:

```python
>>> import amico
>>>
>>> # convert bval/bvec to scheme
>>> amico.util.fsl2scheme("Study01/Subject01/NODDI_protocol.bval", "Study01/Subject01/NODDI_protocol.bvec")
-> Writing scheme file to [ Study01/Subject01/NODDI_protocol.scheme ]
'Study01/Subject01/NODDI_protocol.scheme'

>>> amico.core.setup()
>>>
>>> # set the study and subject directory
>>> ae = amico.Evaluation("Study01", "Subject01")
>>>
>>> # load data
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

>>> # set model for NODDI
>>> ae.set_model("NODDI")
>>>
>>> # generate kernels. You need this only once per study (i.e., same sheme file generates the same kernels).
>>> ae.generate_kernels(lmax = 12)

-> Creating LUT for "NODDI" model:
   [ 149.3 seconds ]                                           @

>>> # load kernels
>>> ae.load_kernels()

-> Resampling LUT for subject "Subject01":
   [ 52.2 seconds ]

>>> # fit the model
>>> ae.fit()

-> Fitting "NODDI" model to 5478 voxels:
   [ 00h 00m 09s ]

>>> # save the results
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

Now you find ODI, , in the subject's subdirectory `AMICO/NODDI`.


```bash
$ ls Study01/Subject01/AMICO/NODDI/
FIT_ICVF.nii.gz		FIT_OD.nii.gz		config.pickle
FIT_ISOVF.nii.gz	FIT_dir.nii.gz
```

Open them with your favorite viewer. Congraturations!!

