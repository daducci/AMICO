# AMICO - python porting

This code implements the AMICO framework in the Python language.

> This is a **work-in-progress** project!

# Installation

## Install dependencies

### Python and DIPY

This version of AMICO is written in [Python](https://www.python.org/) and, internally, it makes use of the [DIPY](http://dipy.org) library.
Please install and configure both Python and DIPY by following the guidelines on the corresponding websites.

### SPArse Modeling Software (SPAMS)

- [Download](http://spams-devel.gforge.inria.fr/downloads.html) the *python interfaces* of the software and follow the instructions provided [here](http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams003.html) to install it.

### Camino toolkit

Depending on the forward-model employed, AMICO can require the [Camino](http://camino.org.uk) toolkit to generate the response functions, e.g. in case of the `Cylinder-Zeppelin-Ball` model.

Please follow the corresponding [documentation](http://cmic.cs.ucl.ac.uk/camino//index.php?n=Main.Installation) to install Camino and make sure to include the folder containing the script `datasynth` in your system path.

### NODDI toolbox

> This implementation in AMICO **does not require** the [NODDI MATLAB toolbox](http://mig.cs.ucl.ac.uk/index.php?n=Download.NODDI) to be present on your system; all the necessary MATLAB functions for generating the response functions of the NODDI model have in fact been ported to Python.


## Install AMICO

Open the system shell, go to the folder containing this file and run:

```bash
pip install .
```

AMICO is now available in your Python interpreter and can be imported as usual:

```python
import amico
```

### Uninstall AMICO

Open the system shell and run:

```bash
pip uninstall amico
```
