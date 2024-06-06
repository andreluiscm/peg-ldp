import persistent as persist

from abc import ABC
import numpy as np


class Database(ABC):
    def __init__(self) -> None:
        self.name = None
        self.data_folder = persist.DATA_FOLDER
        self.delimiter = None
        self.skip_rows = None
        self.database_folder = None
        self.database_file = None
        self.edge_types = None
        self.n_edge_types = None


    def _init_edge_types(self) -> None:
        edge_types = dict()

        for edge_type in range(self.n_edge_types):
            edge_types[np.str_(edge_type)] = edge_type

        return edge_types


# DBLP
# https://arxiv.org/pdf/1612.09435.pdf
# https://github.com/supriya-gdptl/HCODA/tree/master/data
class DBLP(Database):
    def __init__(self) -> None:
        super(DBLP, self).__init__()

        self.name = 'DBLP'
        self.delimiter = ';'
        self.skip_rows = 1
        self.database_folder = 'DBLP'
        self.database_file = 'DBLP.csv'
        self.n_edge_types = 4
        self.edge_types = self._init_edge_types()


# Netscience
# https://manliodedomenico.com/data.php
class Netscience(Database):
    def __init__(self) -> None:
        super(Netscience, self).__init__()

        self.name = 'Netscience'
        self.delimiter = ';'
        self.skip_rows = 1
        self.database_folder = 'netscience'
        self.database_file = 'netscience.csv'
        self.n_edge_types = 13
        self.edge_types = self._init_edge_types()


# Pierre Auger
# https://manliodedomenico.com/data.php
class PierreAuger(Database):
    def __init__(self) -> None:
        super(PierreAuger, self).__init__()

        self.name = 'Pierre Auger'
        self.delimiter = ';'
        self.skip_rows = 1
        self.database_folder = 'pierre_auger'
        self.database_file = 'pierre_auger.csv'
        self.n_edge_types = 16
        self.edge_types = self._init_edge_types()


# Yeast Landscape
# https://manliodedomenico.com/data.php
class YeastLandscape(Database):
    def __init__(self) -> None:
        super(YeastLandscape, self).__init__()

        self.name = 'Yeast Landscape'
        self.delimiter = ';'
        self.skip_rows = 1
        self.database_folder = 'yeast_landscape'
        self.database_file = 'yeast_landscape.csv'
        self.n_edge_types = 4
        self.edge_types = self._init_edge_types()


def init_databases() -> list[Database]:
    databases = [
        DBLP(),
        Netscience(),
        PierreAuger(),
        YeastLandscape()
    ]

    return databases


if __name__ == '__main__':
    databases = init_databases()

    for db in databases:
        print(db)
        print(db.data_folder)
        print(db.database_folder)
        print(db.database_file)
