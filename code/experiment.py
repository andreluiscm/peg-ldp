from differential_impl import *
from graph import *
from log import *
from setup import *
from utils import *
import persistent as persist

from datetime import datetime


class Experiment:
    def __init__(self, *args) -> None:
        approach = args[0]
        assert isinstance(approach, Approach)

        approach_params = args[1]
        assert isinstance(approach_params, dict)

        self.approach = approach
        self.approach_params = approach_params


    def _get_approach_name(self) -> np.str_:
        file_name = f'{self.approach.value}'

        if 'protocols' in self.approach_params.keys():
            file_name += f'#prot'

            for protocol in self.approach_params['protocols']:
                file_name += f'-{protocol.value}'

        if 'edge_heuristic' in self.approach_params.keys():
            file_name += f'#eh-{self.approach_params['edge_heuristic'].value}'

        if 'clustering_heuristic' in self.approach_params.keys():
            file_name += f'#ch-{self.approach_params['clustering_heuristic'].value}'
        
        if 'n_partitions' in self.approach_params.keys():
            file_name += f'#p-{self.approach_params['n_partitions']}'

        if 'n_clusters' in self.approach_params.keys():
            file_name += f'#c-{self.approach_params['n_clusters']}'
        
        if 'top_k_heuristic' in self.approach_params.keys():
            file_name += f'#tkh-{self.approach_params['top_k_heuristic'].value}'

        if 'k' in self.approach_params.keys():
            file_name += f'#k-{self.approach_params['k']}'

        if 'epsilon_allocation' in self.approach_params.keys():
            file_name += f'#ea'

            for allocation in self.approach_params['epsilon_allocation']:
                file_name += f'-{allocation}'

        return file_name


    def _get_dump_file_header(self, execution_params:dict, epsilon:np.int_) -> dict:
        protocols = [protocol.value for protocol in self.approach_params['protocols']] if 'protocols' in self.approach_params.keys() else 'n/a'
        edge_heuristic = self.approach_params['edge_heuristic'].value if 'edge_heuristic' in self.approach_params.keys() else 'n/a'
        clustering_heuristic = self.approach_params['clustering_heuristic'] if 'clustering_heuristic' in self.approach_params.keys() else 'n/a'
        n_partitions = self.approach_params['n_partitions'] if 'n_partitions' in self.approach_params.keys() else 'n/a'
        n_clusters = self.approach_params['n_clusters'] if 'n_clusters' in self.approach_params.keys() else 'n/a'
        top_k_heuristic = self.approach_params['top_k_heuristic'].value if 'top_k_heuristic' in self.approach_params.keys() else 'n/a'
        k = self.approach_params['k'] if 'k' in self.approach_params.keys() else 'n/a'
        epsilon_allocation = self.approach_params['epsilon_allocation'] if 'epsilon_allocation' in self.approach_params.keys() else 'n/a'

        dump_file_header = {
            'database': execution_params['database'].name,
            'approach': self.approach.value,
            'protocol': protocols,
            'edge_heuristic': edge_heuristic,
            'clustering_heuristic': clustering_heuristic,
            'n_partitions': n_partitions,
            'n_clusters': n_clusters,
            'top_k_heuristic': top_k_heuristic,
            'k': k,
            'epsilon': epsilon,
            'epsilon_allocation': epsilon_allocation,
            'n_executions': execution_params['n_executions']
        }

        return dump_file_header


    def execute(self, execution_params:dict) -> None:
        log_msg(f'started experiment execution', True)

        log_msg(f'approach = {self.approach}', True)
        log_msg(f'approach params = {self.approach_params}')
        log_msg(f'execution params = {execution_params}')

        graph = Graph(execution_params['database'])
        graph.print_info()
        
        approach_name = self._get_approach_name()

        for epsilon in execution_params['epsilons']:
            log_msg(f'current epsilon = {epsilon}', True)

            for exec in range(execution_params['n_executions']):
                log_msg(f'current execution = {(exec + 1)}', True)

                dump_file_name = f'experiment_{execution_params['execution_id']}_{approach_name}_{get_datetime()}.csv'
                dump_file_path = f'{persist.RESULTS_FOLDER}/{execution_params['database'].database_folder}/{persist.GRAPHS_FOLDER}/{dump_file_name}'
                dump_file_header = self._get_dump_file_header(execution_params, epsilon)

                approach_execution_params = self.approach_params.copy()
                approach_execution_params['epsilon'] = epsilon
                approach_execution_params['graph'] = graph

                graph_dp = execute(self.approach, approach_execution_params)

                if execution_params['dump_graph']:
                    log_msg(f'started dumping graph', True)
                    graph_dp.dump(dump_file_path, dump_file_header)
                    log_msg(f'finished dumping graph')

        log_msg(f'finished experiment execution', True)
        

def get_datetime() -> np.str_:
     return datetime.now().strftime('%Y%m%d-%H%M%S%f')[:-4]
