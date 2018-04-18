# Removing rician bias before fitting

Inside python, set your study and subject:

```python
import amico
amico.core.setup()
ae = amico.Evaluation("Study02", "Subject01")
```

before loading your data, activate rician debias and set SNR level:

```python
ae.set_config('doDebiasSignal',True)
ae.set_config('DWI-SNR',30.)
```

You can then proceed to load the DWI data, as well as setting the microstructure model to fit as described in the [ActiveAx](https://github.com/davidrs06/AMICO/blob/feature/debias/doc/demos/ACTIVEAX_01.md#load-the-data) and [NODDI](https://github.com/davidrs06/AMICO/blob/feature/debias/doc/demos/NODDI_01.md#load-the-data) tutorials. 

