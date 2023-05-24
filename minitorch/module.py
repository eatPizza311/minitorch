"""A hierarchical data structure stores three things: 1) parameters, 2) user data, 3) other modules."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and othersubmodules. They make up the basis of neural network stacks.

    Attributes:
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        """Initialize the dictionary for tracking parameters and other modules for this instance."""
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all descendent modules to `train`."""
        self.training = True
        for child in self.modules():
            child.train()

    def eval(self) -> None:
        """Set the mode of this module and all descendent modules to `eval`."""
        self.training = False
        for child in self.modules():
            child.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns:
            The name and `Parameter` of each ancestor parameter.
        """
        output = []

        def add_params(name: str, node: Module) -> None:
            """Recurrently adding name-parameter pairs."""
            prefix = f"{name}." if name else ""
            # add current parameters
            for param_name, param in node.__dict__["_parameters"].items():
                output.append((prefix + param_name, param))
            for module_name, module in node.__dict__["_modules"].items():
                add_params(prefix + module_name, module)

        add_params("", self)
        return output

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents."""
        output = []
        for named_parm in self.named_parameters():
            output.append(named_parm[1])
        return output

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
            Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        """Set attribute accordingly."""
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        """Get attribute accordingly."""
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Behavior for using as function."""
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """What Module should look like."""

        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        """Set up parameters profile."""
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        """What Parameter looks like."""
        return repr(self.value)

    def __str__(self) -> str:
        """What Parameter looks like."""
        return str(self.value)
