import json
from typing import List, Dict
from overrides import overrides

from .serializable import Serializable
from .tour import Tour
from .clonable import Clonable


class Solution(Serializable, Clonable):
    """
    A solution consists of planned tours.
    """

    def __init__(self, tours: List[Tour]):
        self._tours = tours

    @property
    def tours(self) -> List[Tour]:
        return self._tours

    def clone(self):
        return Solution([tour.clone() for tour in self.tours])

    @overrides
    def serialize(self) -> Dict[str, object]:
        return dict(
            tours=self.tours
        )


class SolutionEncoder(json.JSONEncoder):
    def default(self, o: Solution):

        if isinstance(o, Serializable):
            return o.serialize()
        else:
            return {'__{}__'.format(o.__class__.__name__): o.__dict__} if hasattr(o, '__dict__') else str(o)
