
from graph import *

from abc import ABC
from enum import Enum


class Approach(Enum):
    APPROACH_RANL = 'approach_ranl'
    APPROACH_CLUSTERING = 'approach_clustering'


class Differential(ABC):
    def __init__(self) -> None:
        pass


    def execute(self, *args) -> Graph:
        pass
