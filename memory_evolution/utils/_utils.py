import inspect


def _function():
    pass


class EmptyDefaultValueError(Exception):
    pass


def get_default_value(func: type(_function), param: str):
    """Get the default value of a function for a given parameter.

    Raises an error if there is no default value for the given parameter.

    Args:
        func: the function
        param: the name of the parameter

    Returns:
        The default value for the given parameter for the given function.

    Raises:
        EmptyDefaultValueError: if there is no default value for the given parameter.
    """
    param = inspect.signature(func).parameters[param]
    if param.default is param.empty:
        raise EmptyDefaultValueError(
            f"function '{func.__module__}.{func.__qualname__}' doesn't have a default value for the param '{param}'.")
    return param.default

