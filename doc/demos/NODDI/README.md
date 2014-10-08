# NODDI fitting tutorial

This tutorial shows how to use the AMICO framework to **fit the NODDI model**, using the example dataset distributed with the [NODDI Matlab Toolbox](http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab).

## Download data for this tutorial

1. Download the original DWI data from [here](http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab).
2. Create the folder `NoddiTutorial/Tutorial` in your data folder and extract into it the content of the downloaded archive `NODDI_example_dataset.zip`.
3. Download the `NODDI_DWI.scheme` scheme file distributed with this tutorial.

## Setup AMICO

Setup the AMICO environment:

```matlab
clearvars, clearvars -global, clc

% Setup AMICO
AMICO_Setup

% Pre-compute auxiliary matrices to speed-up the computations
AMICO_PrecomputeRotationMatrices(); % NB: this needs to be done only once and for all
```

## Load the data

Load the data:

```matlab
% Set the folder containing the data (relative to the data folder).
% This will create a CONFIG structure to keep all the parameters.
AMICO_SetSubject( 'NoddiTutorial', 'Tutorial' );

% Override default file names
CONFIG.dwiFilename    = fullfile( CONFIG.DATA_path, 'NODDI_DWI.hdr' );
CONFIG.maskFilename   = fullfile( CONFIG.DATA_path, 'roi_mask.hdr' );
CONFIG.schemeFilename = fullfile( CONFIG.DATA_path, 'NODDI_DWI.scheme' );

% Load the dataset in memory
AMICO_LoadData
```

The output will look like:

```
-> Loading and setup:
	* Loading DWI...
		- dim    = 128 x 128 x 50 x 81
		- pixdim = 1.875 x 1.875 x 2.500
	* Loading SCHEME...
		- 81 measurements divided in 2 shells (9 b=0)
	* Loading MASK...
		- dim    = 128 x 128 x 50
		- voxels = 5478
   [ DONE ]
```

## Generate the kernels

Generate the kernels corresponding to the different compartments of the NODDI model:

```matlab
% Setup AMICO to use the 'NODDI' model
AMICO_SetModel( 'NODDI' );

% Generate the kernels
AMICO_CreateHighResolutionScheme();
AMICO_GenerateKernels();
AMICO_RotateAndSaveKernels();
```

The output will look something like:

```
-> Create high resolution scheme file for protocol "NoddiTutorial":
   [ DONE ]

-> Simulating high-resolution "NODDI" kernels:
   [ 4.2 seconds ]

-> Rotating kernels to 180x180 directions for subject "Tutorial":
	- A_001.mat... [2.7 seconds]
	- A_002.mat... [2.6 seconds]
	
	...
	
	- A_144.mat... [2.7 seconds]
	- A_145.mat... [0.0 seconds]
	- saving...    [4.3 seconds]
   [ 388.9 seconds ]
```

## Load the kernels

Calculate the kernels corresponding to the different compartments of the NODDI model:

```matlab
% Load the kernels in memory
KERNELS = AMICO_LoadKernels();
```

The output will look like:

```
-> Loading precomputed kernels:
   [ NODDI, 145 atoms, 81 samples ]
```


## Fit the model

Actually **fit** the NODDI model using the AMICO framework:

```matlab
AMICO_fit()
```

The output will look something like:

```
-> Fitting NODDI model to data:
   [ 0h 0m 14s ]

-> Saving output maps:
   [ OUTPUT_*.nii ]
```

![NRMSE for COMMIT](https://github.com/daducci/AMICO/blob/master/doc/demos/NODDI/RESULTS_Fig1.png)

The results will be saved as NIFTI/ANALYZE files in `NoddiTutorial/Tutorial/OUTPUT_*`.


