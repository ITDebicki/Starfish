from collections.abc import Sequence, Mapping, MutableMapping
from typing import List, Optional, Union, Tuple, MutableSet, Iterable
import torch
import numpy as np
import warnings
from .scalers import ParamScaler

def preprocess_dict(d):
    if isinstance(d, Sequence) and isinstance(d[0], Mapping):
        return {str(i): preprocess_dict(d[i]) for i in range(len(d))}
    elif isinstance(d, Mapping):
        return {k: preprocess_dict(v) for k, v in d.items()}
    return d


class GroupedParamDict(MutableMapping):
    """This dictionary stores all values in tensors, so that they can be used in pytorch backpropagation as input parameters. It has behaviour similar to FlatterDict, where keys are seperated and stored at diffferent levels based on the separator, up to a maximum depth.
    Optinoally, the different levels can store the keys in them as a single tensor to allow for easier vectorization of operations.
    The dictionary also supports scalers for parameters, in order to keep parameter values within similar ranges to improve gradient descent performance. They are applied transparently, so no other changes to the code need to be made,
        and only operate internally to scale the raw internal values to the appropriate value when getting or setting the value. 
    """

    def __init__(self, d:Optional[Mapping] = None, separator:str = ':', max_depth:int = -1, groupTensors:Union[bool, List[bool]] = False, error_on_hidden_override:bool = True, device:Union[str, torch.DeviceObjType]='cpu', dtype: Union[str, torch.dtype] = torch.float64):
        """Constructs a new GroupedParamDict.

        Args:
            d (Optional[Mapping], optional): The items to put into the parameter dictionary. Defaults to None.
            separator (str, optional): THe separator to use for different levels of the dictionary. Defaults to ':'.
            max_depth (int, optional): The maximum depth of keys. Defaults to -1.
            groupTensors (Union[bool, List[bool]], optional): If to group tensors at this and future levels of the dictionary into a single vector. Defaults to False.
            error_on_hidden_override (bool, optional): IF to throw an error if a dictionary is overriden with a single value. Defaults to True.
            device (Union[str, torch.DeviceObjType], optional): The device to store the tensors on.. Defaults to 'cpu'.
            dtype (Union[str, torch.dtype], optional): The storage data type to use for the dictionary values , unless a tensor is explicitly passed in when groupig is off. Defaults to torch.float64
        """
        self.separator = separator
        self.max_depth = max_depth
        if isinstance(groupTensors, bool):
            self.groupNestedTensors = groupTensors
        elif len(groupTensors) == 1:
            self.groupNestedTensors = groupTensors[0]
        else:
            self.groupNestedTensors = groupTensors[1:]
        self.error_on_hidden_override = error_on_hidden_override

        self._isGrouped = groupTensors if isinstance(groupTensors, bool) else groupTensors[0]
        self.requires_grad = False

        # Set correct functions depending if it is goruped or single
        self._set_item = self._set_grouped if self._isGrouped else self._set_single
        self._get_item = self._get_item_grouped if self._isGrouped else self._get_item_single
        self._del_item = self._del_item_grouped if self._isGrouped else self._del_item_single
        self.clean = self._clean_grouped if self._isGrouped else self._clean_single
        self._values_f = self._values_grouped if self._isGrouped else self._values_single
        self.params = self._params_grouped if self._isGrouped else self._params_single
        self.to = self._to_grouped if self._isGrouped else self._to_single

        self.device = device
        self.dtype = dtype

        self._frozen = set()

        self._scalers = {}

        # If is grouped, set up the initial tensor values as 0
        if self._isGrouped:
            self._values = torch.zeros(10, dtype = self.dtype)
            self._next = 0
            self._unused_indices = []

        self._d = {}

        # Add items from the initial dictionary to this dictionary
        if d is not None:
            item_stack = list(preprocess_dict(d).items())
            while len(item_stack) > 0:
                k, v = item_stack.pop()
                if isinstance(v, Mapping):
                    item_stack.extend([(k + self.separator + sub_k, sub_v) for sub_k, sub_v in v.items()])
                else:
                    self[k] = v

    def __setitem__(self, key, value):
        key = str(key)
        keyParts = key.split(self.separator, self.max_depth)
        self._set_item_hierchical(keyParts, value)

    def _set_item_hierchical(self, keyParts: List[str], value):
        if len(keyParts) == 1:
            self._set_item(keyParts[0], value)
        else:
            if keyParts[0] in self._d and not isinstance(self._d[keyParts[0]], GroupedParamDict):
                if self.error_on_hidden_override:
                    raise KeyError("Attempted to override single value with grouped dictionary")
                else:
                    warnings.warn("Overwriting value with grouped dictionary")
                del self[keyParts[0]]
            if keyParts[0] not in self._d:
                self._d[keyParts[0]] = GroupedParamDict(
                                                separator = self.separator,
                                                max_depth = self.max_depth - 1,
                                                groupTensors = self.groupNestedTensors,
                                                device = self.device,
                                                dtype = self.dtype)
            self._d[keyParts[0]]._set_item_hierchical(keyParts[1:], value)
                
    def _set_single(self, key: str, value):
        # Convert to tensor
        t = None
        if isinstance(value, torch.Tensor):
            t = value
        elif isinstance(value, np.ndarray) or isinstance(value, Sequence):
            t = torch.tensor(value, dtype = self.dtype)
        else:
            t = torch.tensor([value], dtype = self.dtype)
        # Make sure it is not a 0 size tensor
        t = torch.atleast_1d(t)
        # Copy values if possible, maintaining same tensor, otherwise overwrite
        if key in self._d and isinstance(self._d[key], GroupedParamDict):
            if self.error_on_hidden_override:
                raise KeyError("Attempted to override grouped parameters with single value")
            else:
                warnings.warn("Overwriting grouped parameters with single value")
            del self._d[key]
        # IF requires_grad, then overwrite instead of copy
        if key in self._d and t.shape == self._d[key].shape and not self._d[key].requires_grad:
            # Apply scalers if needed to standardize value
            self._d[key][:] = self._scalers[key].standardize(t) if key in self._scalers else t
        else:
            self._d[key] = self._scalers[key].standardize(t) if key in self._scalers else t

    def _set_grouped(self, key: str, value):
        if key in self._d and isinstance(self._d[key], GroupedParamDict):
            if self.error_on_hidden_override:
                raise KeyError("Attempted to override grouped parameters with single value")
            else:
                warnings.warn("Overwriting grouped parameters with single value")
            del self._d[key]
        if key not in self._d:
            self._d[key] = self._next
            self._next += 1
            if self._next >= len(self._values):
                tmp = torch.zeros(len(self._values) + 10, dtype = self._values.dtype, device = self._values.device)
                tmp[:self._next] = self._values
                self._values = tmp
        self._values[self._d[key]] = self._scalers[key].standardize(value) if key in self._scalers else value

    def __getitem__(self, key) -> torch.Tensor:
        key = str(key)
        keyParts = key.split(self.separator, self.max_depth)
        return self._get_item_hierarchical(keyParts)

    def _get_item_hierarchical(self, keyParts: List[str]) -> torch.Tensor:
        if len(keyParts) == 1:
            return self._get_item(keyParts[0])
        elif isinstance(self._d[keyParts[0]], GroupedParamDict):
            return self._d[keyParts[0]]._get_item_hierarchical(keyParts[1:])
        else:
            raise KeyError("Key not in dictionary")
    
    def _get_item_single(self, key: str) -> torch.Tensor:
        if key in self._scalers:
            return self._scalers[key].original(self._d[key])
        else:
            return self._d[key]

    def _get_item_grouped(self, key: str) -> torch.Tensor:
        if key in self._scalers:
            return self._scalers[key].original(self._values[self._d[key]])
        else:
            return self._values[self._d[key]]

    def __contains__(self, key) -> bool:
        key = str(key)
        return self._contains_hierarchical(key.split(self.separator, self.max_depth))

    def _contains_hierarchical(self, keyParts: List[str]) -> bool:
        if len(keyParts) == 1:
            return keyParts[0] in self._d
        else:
            return keyParts[0] in self._d and isinstance(self._d[keyParts[0]], GroupedParamDict) and self._d[keyParts[0]]._contains_hierarchical(keyParts[1:])

    def __len__(self) -> int:
        return len(self.keys(deep = True))

    def __delitem__(self, key):
        key = str(key)
        keyParts = key.split(self.separator, self.max_depth)
        self._del_item_hierarchical(keyParts)
    
    def _del_item_hierarchical(self, keyParts: List[str]):
        if len(keyParts) == 1:
            self._del_item(keyParts[0])
        elif isinstance(self._d[keyParts[0]], GroupedParamDict):
            deleted = self._d[keyParts[0]]._del_item_hierarchical(keyParts[1:])
            if len(self._d[keyParts[0]]) == 0:
                del self._d[keyParts[0]]
        else:
            raise KeyError("Key not in dictionary")

    def _del_item_single(self, key:str):
        del self._d[key]

    def _del_item_grouped(self, key:str):
        idx = self._d[key]
        self._unused_indices.append(idx)
        del self._d[key]
        
    def _clean_single(self):
        """Cleans any grouped tensors to make them contiguoug=s if any have been deleted."""
        for key, v in self._d.items():
            if isinstance(v, GroupedParamDict):
                v.clean()

    def _clean_grouped(self):
        """Cleans any grouped tensors to make them contiguoug=s if any have been deleted."""
        for idx in sorted(self._unused_indices, reverse = True):
            for key in self._d:
                if not isinstance(self._d[key], GroupedParamDict) and self._d[key] > idx:
                    self._d[key] -= 1
            self._values[idx:-1] = self._values[idx + 1:].clone()
            self._next -= 1
        self._clean_single()

    def values(self, flat:bool = False) -> List[torch.Tensor]:
        """Returns all the values in the dictionary.

        Args:
            flat (bool, optional): If to return all levels of the dictionary or just the current one. Defaults to False.

        Returns:
            List[torch.Tensor]: The list of values in the dictionary.
        """
        if flat:
            keys = self.keys(True)
            return [self[k] for k in keys]
        else:
            return self._values_f()

    def _params_single(self) -> List[torch.Tensor]:
        """Returns list of all not frozen parameters in their raw from. This can be passed to an optimizer as the parameters to optimize"""
        values = []
        for k, v in self._d.items():
            if k not in self._frozen:
                if isinstance(v, GroupedParamDict):
                    values.extend(v.params())
                else:
                    values.append(v)
        return values

    def _params_grouped(self) -> List[torch.Tensor]:
        """Returns list of all not frozen parameters in their raw from. This can be passed to an optimizer as the parameters to optimize"""
        values = [self._values]
        for k, v in self._d.items():
            if isinstance(v, GroupedParamDict) and k not in self._frozen:
                values.extend(v.params())
        return values

    def _values_single(self) -> List[torch.Tensor]:
        values = []
        for k, v in self._d.items():
            if isinstance(v, GroupedParamDict):
                values.extend(v.values())
            else:
                if k in self._scalers:
                    values.append(self._scalers[k].original(v))
                else:
                    values.append(v)
        return values

    def _values_grouped(self) -> List[torch.Tensor]:
        if len(self._unused_indices) > 0:
            self.clean()
        if len(self._scalers) > 0:
            return [self._scalers[k].original(self._values[idx]) if k in self._scalers else self._values[idx] for k, idx in self._d.items() if isinstance(idx, int)] + [v.values() for v in self._d.values() if isinstance(v, GroupedParamDict)]
        else:
            return [self._values[:len(self._d)]] + [v.values() for v in self._d.values() if isinstance(v, GroupedParamDict)]

    def keys(self, deep:bool = False, include_frozen:bool = True) -> List[str]:
        """Gets all the keys present in the dictionary.

        Args:
            deep (bool, optional): IF to include all levels of the dictionary or just this one. Defaults to False.
            include_frozen (bool, optional): If to include keys of parameters that have been frozen. Defaults to True.

        Returns:
            List[str]: The lis tof keys.
        """
        if deep:
            keys = []
            for k in self._d.keys():
                if include_frozen or k not in self._frozen:
                    if isinstance(self._d[k], GroupedParamDict):
                        keys.extend([k + self.separator + k_nested for k_nested in self._d[k].keys(deep, include_frozen)])
                    else:
                        keys.append(k)
            return keys
        else:
            return [k for k in self._d.keys() if include_frozen or k not in self._frozen]

    def items(self, deep:bool = True, include_frozen:bool = True) -> List[Tuple[str, torch.Tensor]]:
        keys = self.keys(deep, include_frozen)
        return [(k, self[k]) for k in keys]

    def __iter__(self):
        return self.keys(True).__iter__()

    def update(self, d: Mapping):
        item_stack = list(preprocess_dict(d).items())
        while len(item_stack) > 0:
            k, v = item_stack.pop()
            if isinstance(v, Mapping):
                item_stack.extend([(k + self.separator + sub_k, sub_v) for sub_k, sub_v in v.items()])
            else:
                self[k] = v

    def frozen(self, deep:bool = True) -> MutableSet[str]:
        """Returns a set of all keys that are frozen.

        Args:
            deep (bool, optional): If to include frozxen keys at lower levels of the dictionary. Defaults to True.

        Returns:
            MutableSet[str]: The set of frozen keys
        """
        return set(self.keys(deep, include_frozen = True)) - set(self.keys(deep, include_frozen = False))

    def freeze(self, keys: Optional[Union[str, Iterable[str]]] = None):
        """Freezes the given key(s) so that they are not modified during backpropagation.

        Args:
            keys (Optional[Union[str, Iterable[str]]], optional): The keys to freeze. If None, freezes all keys. Defaults to None.
        """
        if keys is None:
            keys = self.keys(True)
        elif isinstance(keys, str):
            keys = [keys]
        for k in keys:
            self._freeze_hierarchical(k.split(self.separator, self.max_depth))

    def _freeze_hierarchical(self, keyParts: List[str]):
        if len(keyParts) == 1:
            if keyParts[0] not in self._d:
                pass
                # raise KeyError("Key not present in dictionary")
            elif isinstance(self._d[keyParts[0]], int): # Must be grouped
                raise KeyError("Cannot freeze only one part of parameter group")
            else:
                self._frozen.add(keyParts[0])
                # Disable requires grad
                self._d[keyParts[0]].requires_grad_(False)
        elif keyParts[0] in self._d and isinstance(self._d[keyParts[0]], GroupedParamDict):
            self._d[keyParts[0]]._freeze_hierarchical(keyParts[1:])
        else:
            pass
            # raise KeyError("Key not present in dictionary")

    def thaw(self, keys: Optional[Union[str, Iterable[str]]] = None):
        """Thaws (Unfreezes) the given key(s) so that they are modified during backpropagation.

        Args:
            keys (Optional[Union[str, Iterable[str]]], optional): The keys to thaw. If None, thaws all keys. Defaults to None.
        """
        if keys is None:
            keys = self.keys(True, True)
        elif isinstance(keys, str):
            keys = [keys]
        for k in keys:
            self._thaw_hierarchical(k.split(self.separator, self.max_depth))

    def _thaw_hierarchical(self, keyParts):
        if len(keyParts) == 1:
            if keyParts[0] not in self._d:
                pass
                # raise KeyError("Key not present in dictionary")
            elif isinstance(self._d[keyParts[0]], int): # Must be grouped
                raise KeyError("Cannot thaw only one part of parameter group")
            else:
                if keyParts[0] in self._frozen:
                    self._frozen.remove(keyParts[0])
                self._d[keyParts[0]].requires_grad_(self.requires_grad)
        elif keyParts[0] in self._d and isinstance(self._d[keyParts[0]], GroupedParamDict):
            self._d[keyParts[0]]._thaw_hierarchical(keyParts[1:])
        else:
            pass
            # raise KeyError("Key not present in dictionary")

    def scalers(self) -> Mapping[str, ParamScaler]:
        """
        Returns a dicitonary of all parameter scalers in the dictionary.
        """
        scalers = {}
        for k, scaler in self._scalers.items():
            scalers[k] = scaler
        for k, v in self._d.items():
            if isinstance(v, GroupedParamDict):
                scalers.update(v.scalers())
        return scalers

    def apply_scaler(self, key: Union[Mapping[str, ParamScaler], str], scaler: Optional[ParamScaler] = None):
        """Applies a scaler to the given parameter

        Args:
            key (Union[Mapping[str, ParamScaler], str]): The key to apply the scaler to, or a dictionary of keys to scalers.
            scaler (Optional[ParamScaler], optional): The parameter scaler to use. Defaults to None
        """
        if isinstance(key, str):
            if scaler is None:
                raise ValueError("If key is not a dictionary, must supply a scaler in the scaler argument")
            self._apply_scaler_hierarchical(key.split(self.separator), scaler)
        else:
            for k, scaler in key.items():
                self._apply_scaler_hierarchical(k.split(self.separator), scaler)

    def _apply_scaler_hierarchical(self, keys: List[str], scaler: ParamScaler):
        if len(keys) == 1:
            if keys[0] in self._scalers:
                if keys[0] in self._d:
                    self._d[keys[0]] = self._scalers[keys[0]].original(self._d[keys[0]])
            self._scalers[keys[0]] = scaler
            if keys[0] in self._d:
                self._d[keys[0]] = scaler.standardize(self._d[keys[0]])
        else:
            # Check if exists
            if keys[0] not in self._d:
                self._d[keys[0]] = GroupedParamDict(
                                                separator = self.separator,
                                                max_depth = self.max_depth - 1,
                                                groupTensors = self.groupNestedTensors,
                                                device = self.device)

            self._d[keys[0]]._apply_scaler_hierarchical(keys[1:], scaler)
                

    def remove_scaler(self, key:Union[str, Iterable[str]]):
        """Removes a scaler from a given entry

        Args:
            key (Union[Iterable[str], str]): The key(s) to remove the scaler(s) from
        """
        if isinstance(key, str):
            self._remove_scaler_hierarchical(key.split(self.separator))
        else:
            for k in key:
                self._remove_scaler_hierarchical(k.split(self.separator))
    
    def _remove_scaler_hierarchical(self, keys:List[str]):
        if len(keys) == 1:
            if keys[0] in self._d:
                self._d[keys[0]] = self._scalers[keys[0]].original(self._d[keys[0]])
            del self._scalers[keys[0]]
        else:
            self._d[keys[0]]._remove_scaler_hierarchical(keys[1:])

    def _to_single(self, device):
        self.device = device
        for k in self._d.keys():
            v = self._d[k]
            if isinstance(v, GroupedParamDict):
                v.to(device)
            else:
                self._d[k] = v.to(device)

    def _to_grouped(self, device):
        self.device = device
        self._values = self._values.to(device)
        for k, v in self._d.items():
            if isinstance(v, GroupedParamDict):
                v.to(device)

    def requires_grad_(self, requires_grad):
        [p.requires_grad_(requires_grad) for p in self.params()]
        self.requires_grad = requires_grad

    def to_dict_storage(self) -> dict:
        """Returns a picklable representation of the parameter dictionary for storage"""
        frozen_keys = set(self.keys(deep = True)) - set(set(self.keys(deep = True, include_frozen = False)))
        return {
            'groupTensors': [self._isGrouped] + ([self.groupNestedTensors] if isinstance(self.groupNestedTensors, bool) else self.groupNestedTensors),
            'separator': self.separator,
            'scalers': {k: s.serialize() for k, s in self.scalers().items()},
            'max_depth': self.max_depth,
            'error_on_hidden_override': self.error_on_hidden_override,
            'frozen_keys': frozen_keys,
            'items': {k: v.item() for k, v in self.items()}
        }
        
    @classmethod
    def from_dict_storage(cls, d: Mapping) -> "GroupedParamDict":
        """Creates a new Parameter dictionary from a dictionary returned by "to_dict_storage"

        Args:
            d (Mapping): The dictioanry representation of the parameter dictionary.

        Returns:
            GroupedParamDict: The recreated parameter dict.
        """
        params = cls(d['items'], d['separator'], d['max_depth'], d['groupTensors'], d['error_on_hidden_override'])
        params.freeze(d['frozen_keys'])
        for k, s in d['scalers'].items():
            params.apply_scaler(k, ParamScaler.deserialize(s))
        return params

    def __repr__(self):
        return "<GroupedParamDict {" + ', '.join(["'" + name + "': " + str(val) for name, val in self.items()]) + "}>"