from abc import abstractmethod


class Clonable:
    """
    An object that supports the creation of clones.
    """

    @abstractmethod
    def clone(self):
        """
        :return: A clone of itself.
        """
        pass
