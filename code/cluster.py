import numpy as np


class Cluster:
    def __init__(self, *args) -> None:
        n_args = len(args)

        if n_args == 2:
            id = args[0]
            elements = args[1]

            self.id = id
            self.elements = elements
            self.degrees = None

        elif n_args == 3:
            id = args[0]
            elements = args[1]
            degrees = args[2]

            self.id = id
            self.elements = elements
            self.degrees = degrees

        else:
            raise Exception


    def get_size(self) -> np.int_:
        return len(self.elements)
    

    def get_importance(self) -> np.float_:
        if self.degrees is None:
            raise Exception
        
        else:
            return np.sqrt(np.sum(self.degrees) / self.get_size())
