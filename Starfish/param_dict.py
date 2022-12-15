from collections.abc import Sequence, Mapping, MutableMapping
import torch
import numpy as np

class GroupedParamDict(MutableMapping):

    def __init__(self, d = None, separator = ':', max_depth = -1, groupTensors = False, error_on_hidden_override = True):
        self.separator = separator
        self.max_depth = max_depth
        if isinstance(groupTensors, bool):
            self.groupNestedTensors = groupTensors
        elif len(groupTensors) == 1:
            self.groupNestedTensors = groupTensors[0]
        else:
            self.groupNestedTensors = groupTensors[1:]
        self.error_on_hidden_override = error_on_hidden_override

        isGrouped = groupTensors if isinstance(groupTensors, bool) else groupTensors[0]

        # Set correct functions
        self._set_item = self._set_grouped if isGrouped else self._set_single
        self._get_item = self._get_item_grouped if isGrouped else self._get_item_single
        self._del_item = self._del_item_grouped if isGrouped else self._del_item_single
        self.clean = self._clean_grouped if isGrouped else self._clean_single
        self._values_f = self._values_grouped if isGrouped else self._values_single
        self.params = self._params_grouped if isGrouped else self._params_single

        self._frozen = set()

        if isGrouped:
            self._values = torch.zeros(10, dtype = float)
            self._unused_indices = []

        self._d = {}
        self._size = 0

        if d is not None:
            item_stack = list(d.items())
            while len(item_stack) > 0:
                k, v = item_stack.pop()
                if isinstance(v, Mapping):
                    item_stack.extend([(k + self.separator + sub_k, sub_v) for sub_k, sub_v in v.items()])
                else:
                    self[k] = v

    def __setitem__(self, key, value):
        key = str(key)
        keyParts = key.split(self.separator, self.max_depth)
        self._size += self._set_item_hierchical(keyParts, value)

    def _set_item_hierchical(self, keyParts: str, value):
        if len(keyParts) == 1:
            return self._set_item(keyParts[0], value)
        else:
            if keyParts[0] in self._d and not isinstance(self._d[keyParts[0]], GroupedParamDict):
                if self.error_on_hidden_override:
                    raise KeyError("Attempted to override single value with grouped dictionary")
                else:
                    print("Overwriting value with grouped dictionary")
                del self[keyParts[0]]
            if keyParts[0] not in self._d:
                self._d[keyParts[0]] = GroupedParamDict(
                                                separator = self.separator,
                                                max_depth = self.max_depth - 1,
                                                groupTensors = self.groupNestedTensors)
            return self._d[keyParts[0]]._set_item_hierchical(keyParts[1:], value)
                
    def _set_single(self, key, value):
        # Convert to tensor
        t = None
        if isinstance(value, torch.TensorType):
            t = value
        elif isinstance(value, np.ndarray) or isinstance(value, Sequence):
            t = torch.DoubleTensor(value)
        else:
            t = torch.DoubleTensor([value])
        # Copy values if possible, maintaining same tensor, otherwise overwrite
        expanded = 0
        if key in self._d and isinstance(self._d[key], GroupedParamDict):
            if self.error_on_hidden_override:
                raise KeyError("Attempted to override grouped parameters with single value")
            else:
                print("Overwriting grouped parameters with single value")
            del self._d[key]
        if key in self._d and t.shape == self._d[key].shape:
            self._d[key][:] = t
            expanded = 1
        else:
            expanded = 1 if key in self._d else 0
            self._d[key] = t
        return expanded

    def _set_grouped(self, key, value):
        expanded = 0
        if key in self._d and isinstance(self._d[key], GroupedParamDict):
            if self.error_on_hidden_override:
                raise KeyError("Attempted to override grouped parameters with single value")
            else:
                print("Overwriting grouped parameters with single value")
            del self._d[key]
        if key not in self._d:
            self._d[key] = len(self._d)
            if len(self._d) >= len(self._values):
                tmp = torch.zeros(len(self._values) + 10, dtype = float)
                tmp[:len(self._d)] = self._values
                self._values = tmp
            expanded = 1
        self._values[self._d[key]] = value
        return expanded

    def __getitem__(self, key):
        key = str(key)
        keyParts = key.split(self.separator, self.max_depth)
        return self._get_item_hierarchical(keyParts)

    def _get_item_hierarchical(self, keyParts):
        if len(keyParts) == 1:
            return self._get_item(keyParts[0])
        elif isinstance(self._d[keyParts[0]], GroupedParamDict):
            return self._d[keyParts[0]]._get_item_hierarchical(keyParts[1:])
        else:
            raise KeyError("Key not in dictionary")
    
    def _get_item_single(self, key):
        return self._d[key]

    def _get_item_grouped(self, key):
        return self._values[self._d[key]]

    def __contains__(self, key):
        key = str(key)
        return self._contains_hierarchical(key.split(self.separator, self.max_depth))

    def _contains_hierarchical(self, keyParts):
        if len(keyParts) == 1:
            return keyParts[0] in self._d
        else:
            return keyParts[0] in self._d and isinstance(self._d[keyParts[0]], GroupedParamDict) and self._d._contains_hierarchical(keyParts[1:])

    def __len__(self):
        return self._size

    def __delitem__(self, key):
        key = str(key)
        keyParts = key.split(self.separator, self.max_depth)
        self._size -= self._del_item_hierarchical(keyParts)
    
    def _del_item_hierarchical(self, keyParts):
        if len(keyParts) == 1:
            return self._del_item(keyParts[0])
        elif isinstance(self._d[keyParts[0]], GroupedParamDict):
            deleted = self._d[keyParts[0]]._del_item_hierarchical(keyParts[1:])
            if len(self._d[keyParts[0]]) == 0:
                del self._d[keyParts[0]]
            return deleted
        else:
            raise KeyError("Key not in dictionary")

    def _del_item_single(self, key):
        del self._d[key]
        return 1

    def _del_item_grouped(self, key):
        idx = self._d[key]
        self._unused_indices.append(idx)
        del self._d[key]
        return 1
        
    def _clean_single(self):
        for key, v in self._d.items():
            if isinstance(v, GroupedParamDict):
                v.clean()

    def _clean_grouped(self):
        for idx in sorted(self._unused_indices, reverse = True):
            for key in self._d:
                if not isinstance(self._d[key], GroupedParamDict) and self._d[key] > idx:
                    self._d[key] -= 1
            self._values[idx:-1] = self._values[idx + 1:].clone()
        self._clean_single()

    def values(self, flat = False):
        if flat:
            keys = self.keys(True)
            return [self[k] for k in keys]
        else:
            return self._values_f()

    def _params_single(self):
        values = []
        for k, v in self._d.items():
            if k not in self._frozen:
                if isinstance(v, GroupedParamDict):
                    values.extend(v.params())
                else:
                    values.append(v)
        return values

    def _params_grouped(self):
        values = [self._values]
        for k, v in self._d.items():
            if isinstance(v, GroupedParamDict) and k not in self._frozen:
                values.extend(v.params())
        return values

    def _values_single(self):
        values = []
        for v in self._d.values():
            if isinstance(v, GroupedParamDict):
                values.extend(v.values())
            else:
                values.append(v)
        return values

    def _values_grouped(self):
        if len(self._unused_indices) > 0:
            self.clean()
        return [self._values[:len(self._d)]] + [v.values() for v in self._d.values() if isinstance(v, GroupedParamDict)]

    def keys(self, deep = False):
        if deep:
            keys = []
            for k in self._d.keys():
                if isinstance(self._d[k], GroupedParamDict):
                    keys.extend([k + self.separator + k_nested for k_nested in self._d[k].keys(True)])
                else:
                    keys.append(k)
            return keys
        else:
            return self._d.keys()

    def items(self):
        keys = self.keys(True)
        return [(k, self[k]) for k in keys]

    def __iter__(self):
        return self.keys(True).__iter__()

    def freeze(self, keys = None):
        if keys is None:
            keys = self.keys(True)
        elif isinstance(keys, str):
            keys = [keys]
        for k in keys:
            self._freeze_hierarchical(k.split(self.separator, self.max_depth))

    def _freeze_hierarchical(self, keyParts):
        if len(keyParts) == 1:
            if keyParts[0] not in self._d:
                raise KeyError("Key not present in dictionary")
            elif isinstance(self._d[keyParts[0]], int): # Must be grouped
                raise KeyError("Cannot freeze only one part of parameter group")
            else:
                self._frozen.add(keyParts[0])
        elif keyParts[0] in self._d and isinstance(self._d[keyParts[0]], GroupedParamDict):
            self._d[keyParts[0]]._freeze_hierarchical(keyParts[1:])
        else:
            raise KeyError("Key not present in dictionary")

    def thaw(self, keys = None):
        if keys is None:
            keys = self.keys(True)
        elif isinstance(keys, str):
            keys = [keys]
        for k in keys:
            self._thaw_hierarchical(k.split(self.separator, self.max_depth))

    def _thaw_hierarchical(self, keyParts):
        if len(keyParts) == 1:
            if keyParts[0] not in self._d:
                raise KeyError("Key not present in dictionary")
            elif isinstance(self._d[keyParts[0]], int): # Must be grouped
                raise KeyError("Cannot freeze only one part of parameter group")
            else:
                self._frozen.remove(keyParts[0])
        elif keyParts[0] in self._d and isinstance(self._d[keyParts[0]], GroupedParamDict):
            self._d[keyParts[0]]._freeze_hierarchical(keyParts[1:])
        else:
            raise KeyError("Key not present in dictionary")

    def requires_grad_(self, requires_grad):
        [p.requires_grad_(requires_grad) for p in self.params()]

    def __repr__(self):
        return "<GroupedParamDict {" + ', '.join(["'" + name + "': " + str(val) for name, val in self.items()]) + "}>"