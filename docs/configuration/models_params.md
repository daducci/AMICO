# Models parameters
You can set the model parameters by calling the `set()` method of the `model` attribute of the `Evaluation` class:
```Python
>>> import amico

>>> ae = amico.Evaluation()
>>> ae.set_model('NODDI')
>>> ae.model.set(param1=value1, param2=value2, ...)
```

Here is a list of the parameters that can be set for each model:
