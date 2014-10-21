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

Add the folder containing the source code of AMICO to your `MATLAB PATH`.

Copy the file `AMICO_Setup.txt` and rename it to `AMICO_Setup.m`. Modify its content to set the paths to your specific needs:

- `AMICO_code_path` : path to the folder containing the source code of AMICO (this repository). E.g. `/home/user/AMICO/code`.

- `CAMINO_path` : path to the `bin` folder containing the executables of the Camino toolkit (in case you want to use ActiveAx, not needed for NODDI). E.g. `/home/user/camino/bin`.

- `AMICO_data_path` : path to the folder where you store all your datasets. E.g. `/home/user/AMICO/data`. Then, the software assumes the folder structure is the following:
    ```
    ├── data
        ├── Study_01                 --> all subjects acquired with protocol "Study_01"
            ├── Subject_01
            ├── Subject_02
            ├── ...
        ├── Study_02                 --> all subjects acquired with protocol "Study_02"
            ├── Subject_01
            ├── Subject_02
            ├── ...
        ├── ...
    ```
  This way, the kernels need to be computed only *once per each study*, i.e. same protocol (number of shells, b-values etc), and subsequently adapted to each subject (specific gradient directions) very efficiently.


## Getting started

Tutorials/demos are provided in the folder `doc/demos` to help you get started with the AMICO framework.
