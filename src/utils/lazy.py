# pylint: disable=unused-argument
'''Lazy imports utilities.'''

from __future__ import annotations

import importlib
import threading
from typing import Any, Callable

# -----------------------------------------------------------------------------
# Lazy function loader
# -----------------------------------------------------------------------------
def lazy_function(module_path: str, func_name: str) -> Callable:
    '''
    A lazy function that loads the real implementation when called.
    '''
    def _wrapper(*args, **kwargs):
        module = __import__(module_path, fromlist=[func_name])
        func = getattr(module, func_name)
        return func(*args, **kwargs)
    _wrapper.__name__ = func_name
    _wrapper.__doc__ = f'Lazy-loaded function {module_path}.{func_name}'
    return _wrapper


# -----------------------------------------------------------------------------
# Lazy class proxy
# -----------------------------------------------------------------------------

class LazyClassProxy:
    '''
    Proxy that behaves like a class but loads the real class lazily.

    * Internal attributes are always accessed via object.__getattribute__/__setattr__
      to avoid routing through __getattr__ and causing recursion.
    * A reentrant lock prevents double initialization under concurrency.
    * __wrapped__ is set so tools (inspect, help) can discover the underlying class.
    '''

    __slots__ = ('_module', '_name', '_cached', '_lock', '__wrapped__')

    def __init__(self, module_path: str, class_name: str) -> None:
        object.__setattr__(self, '_module', module_path)
        object.__setattr__(self, '_name', class_name)
        object.__setattr__(self, '_cached', None)
        object.__setattr__(self, '_lock', threading.RLock())
        object.__setattr__(self, '__wrapped__', None)

    def _load(self) -> type:
        cached = object.__getattribute__(self, '_cached')
        if cached is not None:
            return cached

        lock = object.__getattribute__(self, '_lock')
        with lock:
            cached = object.__getattribute__(self, '_cached')
            if cached is None:
                module_path = object.__getattribute__(self, '_module')
                class_name = object.__getattribute__(self, '_name')
                mod = importlib.import_module(module_path)
                real = getattr(mod, class_name)
                object.__setattr__(self, '_cached', real)
                object.__setattr__(self, '__wrapped__', real)
                return real
            return cached

    # Behave like the real class
    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        # Only called if normal attribute lookup fails; forward to real class.
        real = self._load()
        return getattr(real, item)

    def __instancecheck__(self, instance) -> bool:  # for isinstance(proxy, T)
        return isinstance(instance, self._load())

    def __subclasscheck__(self, subclass) -> bool:  # for issubclass(proxy, T)
        return issubclass(subclass, self._load())

    def __mro_entries__(self, bases):
        return (self._load(),)

    def __repr__(self) -> str:
        module_path = object.__getattribute__(self, '_module')
        class_name = object.__getattribute__(self, '_name')
        return f'<LazyClassProxy {module_path}.{class_name}>'
