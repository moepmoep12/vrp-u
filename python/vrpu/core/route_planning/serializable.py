import json
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


class JSONSerializer(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, Serializable):
            return o.serialize()
        else:
            return {'__{}__'.format(o.__class__.__name__): o.__dict__} if hasattr(o, '__dict__') else str(o)
