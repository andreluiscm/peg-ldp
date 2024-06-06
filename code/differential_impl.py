from approach_ranl import *
from approach_clustering import *


def execute(approach:Approach, args:dict) -> Graph:
    if approach == Approach.APPROACH_RANL:
        return Approach_RANL().execute(args['graph'], args['protocols'], args['edge_heuristic'], args['epsilon'], args['epsilon_allocation'])
    
    elif approach == Approach.APPROACH_CLUSTERING:
        return Approach_Clustering().execute(args['graph'], args['protocols'], args['edge_heuristic'],
                                             args['clustering_heuristic'], args['n_partitions'], args['n_clusters'],
                                             args['top_k_heuristic'], args['k'], args['epsilon'], args['epsilon_allocation'])
    
    else:
        raise Exception
