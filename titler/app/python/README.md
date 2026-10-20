# titler for Python

A 4 MB model that turns a prompt into a short title. It runs on the CPU in about 15-100 ms.

```python
from titler import Titler

t = Titler()                      # the model inside the wheel
print(t.title("how do I sort a list of dicts by a key in python"))   # Sorting Dicts By Key in Python
```

`Titler("path/to/model.gguf")` loads another model file. `t.title(text, beams=4)`: 1 beam is fastest, 4 is the default.
One `Titler` can be used from many threads at the same time.

This package is a thin layer over the C library in the same repository (`app/capi`).
