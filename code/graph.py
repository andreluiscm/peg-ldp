from databases import *

import numpy as np


class Graph:
    def __init__(self, *args) -> None:
        n_args = len(args)

        if n_args == 1:
            db = args[0]

            assert isinstance(db, Database)

            self.name = db.name
            self.adj_matrix = self._load_graph(db)
            self.n_nodes = self.adj_matrix.shape[0]
            self.n_edges = self._get_num_edges()
            self.edge_types = db.edge_types
            self.n_edge_types = db.n_edge_types

        elif n_args == 3:
            name = args[0]
            adj_matrix = args[1]
            edge_types = args[2]

            assert isinstance(name, str)
            assert isinstance(adj_matrix, np.ndarray)
            assert isinstance(edge_types, dict)

            self.name = name
            self.adj_matrix = adj_matrix
            self.n_nodes = adj_matrix.shape[0]
            self.n_edges = self._get_num_edges()
            self.edge_types = edge_types
            self.n_edge_types = len(edge_types)
            self.is_baseline = None

        else:
            raise Exception


    def _load_graph(self, db:Database) -> np.ndarray:
        graph_file = f'{db.data_folder}/{db.database_folder}/{db.database_file}'

        data = np.loadtxt(
            fname=graph_file,
            dtype=np.str_,
            delimiter=db.delimiter,
            skiprows=db.skip_rows)

        n_nodes = np.maximum(np.max(data[:, 0].astype(np.int_)), np.max(data[:, 1].astype(np.int_))) + 1
        adj_matrix = np.full((n_nodes, n_nodes, db.n_edge_types), False, dtype=np.bool_)

        for row in data:
            i = np.int_(row[0])
            j = np.int_(row[1])
            edge_type_id = db.edge_types[row[2]]

            adj_matrix[i, j, edge_type_id] = True
            adj_matrix[j, i, edge_type_id] = True

        return adj_matrix
    

    def _get_num_edges(self) -> np.int_:
        return np.int_(np.sum(self.adj_matrix) / 2)
        

    def update_num_edges(self) -> None:
        self.n_edges = self._get_num_edges()

    
    def get_degrees(self) -> np.ndarray:
        return np.sum(np.sum(self.adj_matrix, axis=1), axis=1).astype(np.int_)
    
    
    def get_degree_sequence(self, is_ascending:np.bool_) -> np.ndarray:
        if is_ascending:
            return np.sort(self.get_degrees())
        else:
            return np.sort(self.get_degrees())[::-1]


    def get_degrees_by_edge_type(self) -> np.ndarray:
        return np.sum(self.adj_matrix, axis=1).astype(np.int_)
    

    def dump(self, path:np.str_, header:dict) -> None:
        edges = np.vstack((np.where(self.adj_matrix == 1))).T[:self.n_edges]
        np.savetxt(fname=path, X=edges, fmt='%d', delimiter=';', header=np.str_(header), comments='')


    def print_info(self) -> None:
        print(f'\n-- GRAPH INFO --')
        print(f'.')
        print(f'name = {self.name}')
        print(f'# nodes = {self.n_nodes}')
        print(f'# edges = {self.n_edges}')
        print(f'edge types = {self.edge_types}')
        print(f'# edge types = {self.n_edge_types}')
        print(f'.')


if __name__ == '__main__':
    db = PierreAuger()

    g = Graph(db)
    g.print_info()
