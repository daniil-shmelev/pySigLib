# pysiglib-cuda

CUDA plugin for [pysiglib](https://github.com/daniil-shmelev/pySigLib). Ships the `cusig` shared library as a sibling package that `pysiglib` discovers at import time.

Install via the `[cuda]` extra on pysiglib:

```bash
pip install pysiglib[cuda]
```

Do not install this package standalone - it depends on a matching version of `pysiglib`.
