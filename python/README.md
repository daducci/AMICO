# AMICO - python porting

This code implements the AMICO framework in the Python language.

> This is a **working progress** project!

# Installation

## Install dependencies

### Python and DIPY

AMICO is written in [Python](https://www.python.org/) and it internally makes use of the [DIPY](http://dipy.org) library.
Please install and configure them by following the guidelines on the corresponding websites.

### SPArse Modeling Software (SPAMS)

- [Download](http://spams-devel.gforge.inria.fr/downloads.html) the *python interfaces* of the software and follow the instructions provided [here](http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams003.html) to install it.  

### Camino toolkit

Depending on the forward-model employed, AMICO can require the [Camino](http://camino.org.uk) toolkit to generate the response functions, e.g. in case of the `Cylinder-Zeppelin-Ball` model.

Please follow the corresponding [documentation](http://cmic.cs.ucl.ac.uk/camino//index.php?n=Main.Installation) to install Camino and make sure to include the folder containing the script `datasynth` in your system path.

### NODDI toolbox


> We are in the process of porting to Python all the MATLAB functions that are required to generate the response functions of the NODDI model. Hence, at the moment the **NODDI model is not available**, sorry...



## Install AMICO

Open the system shell, go to the folder containing this file and run:

```bash
python setup.py install
```

AMICO is now available in your Python interpreter and can be imported as usual:

```python
import amico
```
