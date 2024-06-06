from experiment import *
from setup import *


if __name__ == '__main__':
    setup = Setup()
    setup.print_info()

    exp_1 = Experiment(Approach.APPROACH_RANL, {'protocols': [ProtocolInit.RR], 'edge_heuristic': EdgeConsistencyHeuristic.RANDOM, 'epsilon_allocation': [100]})

    exp_2 = Experiment(Approach.APPROACH_RANL, {'protocols': [ProtocolInit.RR], 'edge_heuristic': EdgeConsistencyHeuristic.AND, 'epsilon_allocation': [100]})

    exp_3 = Experiment(Approach.APPROACH_CLUSTERING, {'protocols': [ProtocolInit.OUE, ProtocolInit.RR], 'edge_heuristic': EdgeConsistencyHeuristic.AND,
                                                      'clustering_heuristic': ClusteringHeuristic.FIXED_RANDOM, 'n_partitions': None, 'n_clusters': None,
                                                      'top_k_heuristic':TopKHeuristic.FIXED, 'k':1, 'epsilon_allocation': [20, 80]})
    
    exp_4 = Experiment(Approach.APPROACH_CLUSTERING, {'protocols': [ProtocolInit.OUE, ProtocolInit.RR], 'edge_heuristic': EdgeConsistencyHeuristic.AND,
                                                      'clustering_heuristic': ClusteringHeuristic.DYNAMIC_DEGREE, 'n_partitions': None, 'n_clusters': None,
                                                      'top_k_heuristic':TopKHeuristic.COUNT_PERCENTIL, 'k':70, 'epsilon_allocation': [20, 20, 60]})
    
    experiments = [exp_1, exp_2, exp_3, exp_4]

    for db in setup.databases:
        log_msg(f'current database = {db.database_folder}', True)

        exec_id = persist.get_execution_id(db.database_folder)

        execution_params = params = {
            'database': db,
            'execution_id': exec_id,
            'epsilons': setup.epsilons,
            'n_executions': setup.n_executions,
            'dump_graph': setup.dump_graph
        }

        n_partitions = setup.part_clus_pairs[db.database_folder][0]
        n_clusters = setup.part_clus_pairs[db.database_folder][1]

        exp_3.approach_params['n_partitions'] = n_partitions
        exp_3.approach_params['n_clusters'] = n_clusters
        
        exp_4.approach_params['n_partitions'] = n_partitions
        exp_4.approach_params['n_clusters'] = n_clusters
        
        for experiment in experiments:
            experiment.execute(execution_params)
