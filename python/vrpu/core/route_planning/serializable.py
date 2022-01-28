from abc import abstractmethod
from typing import Dict


class Serializable(object):
    """
    Interface for a serializable object.
    """

    @abstractmethod
    def serialize(self) -> Dict[str, object]:
        """
        Serializes this object.
        :return: Dictionary for serialized properties.
        """
        pass
