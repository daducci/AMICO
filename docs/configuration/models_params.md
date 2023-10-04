# Models parameters
You can set the model parameters by calling the `set()` method of the `model` attribute of the `Evaluation` class:
```python
>>> import amico

>>> ae = amico.Evaluation()
>>> ae.set_model('NODDI')
>>> ae.model.set(param1=value1, param2=value2, ...)
```

See the [models api reference](../api_reference/models.md) for more information about the available parameters for each model.
