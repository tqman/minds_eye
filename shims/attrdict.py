"""Minimal stand-in for the unmaintained ``attrdict`` package.

The upstream package does ``from collections import Mapping`` and cannot be
imported on Python >= 3.10. Janus only uses ``AttrDict`` as a dict whose keys
are also attributes (config params), which is all this provides.
"""


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                self[k] = AttrDict(v)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value
