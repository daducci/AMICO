# Decreasing Resolution

This tutorial illustrates how to **drecrease the resolution of the look-up table (a.k.a LUT)** in AMICO.

## Starting Script

Check the **NODDI_01** demo from the following [link](https://github.com/daducci/AMICO/blob/master/doc/demos/NODDI_01.md), which contains an example dataset and the following code:

```python
import amico

amico.core.setup()

ae = amico.Evaluation("Study01", "Subject01")

amico.util.fsl2scheme("Study01/Subject01/NODDI_protocol.bval", "Study01/Subject01/NODDI_protocol.bvec")

ae.load_data(dwi_filename = "NODDI_DWI.img", scheme_filename = "NODDI_protocol.scheme", mask_filename = "roi_mask.img", b0_thr = 0)

ae.set_model("NODDI")
ae.generate_kernels()

ae.load_kernels()

ae.fit()

ae.save_results()
```

## Change the resolution

In order to change the resolution of the LUT in AMICO, we have to specify the desired number of directions by changing the value of the parameter `ndirs` in the functions:

- amico.core.setup()
- ae.generate_kernels()

In this example, we modify the above code to use 500 directions. The modified code should be like this:

```python
import amico

amico.core.setup( ndirs=500 )

ae = amico.Evaluation("Study01", "Subject01")

amico.util.fsl2scheme("Study01/Subject01/NODDI_protocol.bval", "Study01/Subject01/NODDI_protocol.bvec")

ae.load_data(dwi_filename = "NODDI_DWI.img", scheme_filename = "NODDI_protocol.scheme", mask_filename = "brain_mask.img", b0_thr = 0)

ae.set_model("NODDI")
ae.generate_kernels( ndirs=500 )

ae.load_kernels()

ae.fit()

ae.save_results()
```

Please note that the value of `ndirs` **must match around all the functions**. When the value of `ndirs` is not specified, AMICO will use as default `ndirs = 181*181 = 32,761` directions per shell. AMICO does not support an arbitrary number of directions. The value of `ndirs` has to be one the values in the set: {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 32761 (default)}.

## Differences in Memory

The following figure shows the differences in memory of the LUT.

```
|    ndirs value    |    memory usage   |
|        500        |   50   Megabytes  |
|        1000       |   100  Megabytes  |
|        1500       |   150  Megabytes  |
|        2000       |   200  Megabytes  |
|        2500       |   250  Megabytes  |
|        3000       |   300  Megabytes  |
|        3500       |   350  Megabytes  |
|        4000       |   400  Megabytes  |
|        4500       |   450  Megabytes  |
|        5000       |   500  Megabytes  |
|        5500       |   550  Megabytes  |
|        6000       |   600  Megabytes  |
|        6500       |   650  Megabytes  |
|        7000       |   700  Megabytes  |
|        7500       |   750  Megabytes  |
|        8000       |   800  Megabytes  |
|        8500       |   850  Megabytes  |
|        9000       |   900  Megabytes  |
|        9500       |   950  Megabytes  |
|       10000       |     1  Gigabytes  |
|       32761       |   3.2  Gigabytes  |
```

## Differences in the Fitting

A change in the resolution affects the fitting process. The following figures illustrates the differences in the fitting between `ndirs=500` and `ndirs=32761`.

![Track-density](https://github.com/daducci/AMICO/blob/master/doc/demos/fig_comparison_resolution_LUT.png)