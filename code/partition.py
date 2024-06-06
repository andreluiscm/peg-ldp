from cluster import *

import numpy as np


class Partition:
    def __init__(self, id, elements) -> None:
        self.id = id
        self.elements = elements
        self.clusters = None

    def get_size(self) -> np.int_:
        return len(self.elements)
    

    def set_clusters(self, clusters:np.ndarray) -> None:
        self.clusters = clusters


    def get_clusters_size(self) -> np.int_:
        size = 0

        for cluster in self.clusters:
            size += cluster.get_size()

        return size
    

    def get_clusters_elements(self) -> np.ndarray:
        elements = np.array([], dtype=np.int_)

        for cluster in self.clusters:
            elements = np.append(elements, cluster.elements)

        return elements
