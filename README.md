# AMICO

Implementation of the linear framework for Accelerated Microstructure Imaging via Convex Optimization (AMICO) described here:

> **Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data**
>*Alessandro Daducci, Erick J. Canales-Rodriguez, Hui Zhang, Tim B. Dyrby, Daniel C. Alexander, Jean-Philippe Thiran*
>, NeuroImage, 2014 (in press)

## Installation

### Download and install external software

- **NODDI MATLAB toolbox**. [Download](http://mig.cs.ucl.ac.uk/index.php?n=Download.NODDI) the software and follow the instructions provided [here](http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab) to install it. Be sure to properly include this toolbox in your `MATLAB PATH`.

- **CAMINO toolkit**. [Download](http://cmic.cs.ucl.ac.uk/camino//index.php?n=Main.Download) the software and follow the instructions provided [here](http://cmic.cs.ucl.ac.uk/camino//index.php?n=Main.Installation) to install it. NB: be sure to properly update the configuration variable `CAMINO_path` (see later).

- **SPArse Modeling Software**. [Download](http://spams-devel.gforge.inria.fr/downloads.html) the software and follow the instructions provided [here](http://spams-devel.gforge.inria.fr/doc/html/doc_spams003.html) to install it. Be sure to properly include this toolbox in your `MATLAB PATH`.

### Setup paths/variables in MATLAB

Add the folder containing the AMICO code to your `MATLAB PATH`.

Copy the file `AMICO_Setup.txt` and rename it to `AMICO_Setup.m`. Modify its content to set the paths to your specific needs, eg `CAMINO_path`.

## Getting started

Tutorials/demos are provided in the folder `doc/demos` to help you get started with the AMICO framework.
