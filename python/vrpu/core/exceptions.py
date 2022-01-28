class WrongStrictTypeException(Exception):
    """
    Error when receiving different type than expected. No subtype allowed.
    """

    def __init__(self, expected: str, actual: str):
        super().__init__(f"Expected type '{expected}' but got '{actual}' instead.")


class WrongSubTypeException(Exception):
    """
    Error when receiving different type than expected. Subtype allowed.
    """

    def __init__(self, expected: str, actual: str):
        super().__init__(f"Expected sub-type of '{expected}' but got '{actual}' instead.")


class NoneValueException(Exception):
    """
    Error when a variable is none.
    """

    def __init__(self, variable_name: str):
        super().__init__(f"Variable '{variable_name}' is None.")


class EmptyCollectionException(Exception):
    """
    Error when a collection is empty.
    """

    def __init__(self, variable_name: str):
        super().__init__(f"Collection '{variable_name}' is empty.")


class OutOfRangeException(Exception):
    """
    Error when a variable is out of range.
    """

    def __init__(self, min_val: int, max_val: int, actual: int):
        super().__init__(f"Index {actual} is out of range [{min_val},{max_val}].")
