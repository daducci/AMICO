# AMICO

Implementation of the linear framework for Accelerated Microstructure Imaging via Convex Optimization (AMICO) described here:

> **Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data**  
> *Alessandro Daducci, Erick Canales-Rodriguez, Hui Zhang, Tim Dyrby, Daniel Alexander, Jean-Philippe Thiran*  
> NeuroImage 105, pp. 32-44 (2015)

## Code implementation

This is the current implementation of the AMICO framework and it is written in **python**.

# Installation

## Dependencies

This version of AMICO is written in [Python](https://www.python.org/) and, internally, it makes use of the [DIPY](http://dipy.org) library.

AMICO fits the models to the data using the [SPArse Modeling Software (SPAMS)](http://spams-devel.gforge.inria.fr).

Depending on the forward-model employed, AMICO can require the [Camino](http://camino.org.uk) toolkit to generate the response functions, e.g. in case of the `Cylinder-Zeppelin-Ball` model. Please follow the corresponding [documentation](http://cmic.cs.ucl.ac.uk/camino//index.php?n=Main.Installation) to install Camino and make sure to include the folder containing the script `datasynth` in your system path.

> NB: this implementation of AMICO **does not require** the [NODDI MATLAB toolbox](http://mig.cs.ucl.ac.uk/index.php?n=Download.NODDI) to be present on your system; all the necessary MATLAB functions for generating the response functions of the NODDI model have in fact been ported to Python.


## Install AMICO

Open the system shell and run:

```bash
pip install dmri-amico
```

This will download and install AMICO from the [Python Package Index](https://pypi.org).

AMICO is now available in your Python interpreter and can be imported as usual:

```python
import amico
```

### Uninstall AMICO

Open the system shell and run:

```bash
pip uninstall dmri-amico
```
