# Removing rician bias before fitting
In a Python shell, import the `AMICO` library and instantiate an `Evaluation` object:
```python
>>> import amico

>>> ae = amico.Evaluation()
```

Before loading your data, activate Rician debias and set the signal-to-noise ratio (SNR) level with the `set_config()` method:
```python
>>> ae.set_config('doDebiasSignal', True)
>>> ae.set_config('DWI-SNR', 30.0)
```

You can then proceed loading your DWI data, as well as setting the microstructure model to fit.
