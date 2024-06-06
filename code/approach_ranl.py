from differential import *
from frequency_oracle import *
from log import *
from protocols import *
from utils import *


#####################
##  Approach RANL  ##
#####################

class Approach_RANL(Differential):
    def __init__(self) -> None:
        super(Approach_RANL, self).__init__()


    def execute(self, graph:Graph, protocols:list[ProtocolInit], heuristic:EdgeConsistencyHeuristic, epsilon:np.float_, epsilon_allocation:list[np.int_]) -> Graph:
        log_msg(f'started approach_1 execution', True)

        # Ensuring the use of 100% of the privacy budget
        assert np.sum(epsilon_allocation) == 100

        # Ensuring the correct number of epsilon parts
        assert len(epsilon_allocation) == 1

        epsilon_ranl = epsilon * epsilon_allocation[0] / 100

        # Step 1: perturb and aggregate (RANL)
        log_msg(f'started step 1: perturb and aggregate (RANL)')
        n = graph.n_nodes
        d = graph.n_nodes * graph.n_edge_types
        
        protocol_ranl = init_protocol(protocols[0], n, d, epsilon_ranl)
        fo_ranl = FrequencyOracleProtocol(protocol_ranl.n, protocol_ranl.d, protocol_ranl.p, protocol_ranl.q)

        for adj_matrix_i in graph.adj_matrix:
            ranl = get_RANL_from_adj_matrix(adj_matrix_i)
            ranl_p = protocol_ranl.perturb(ranl)

            fo_ranl.aggregate(ranl_p)
        
        fo_ranl.convert_responses_to_array()
        log_msg(f'finished step 1')

        # Step 2: generate DP graph from perturbed RANLs
        log_msg(f'started step 2: generate DP graph from perturbed RANLs')
        adj_matrix_dp = build_graph_from_RANLs(fo_ranl.responses, graph.n_nodes, graph.n_edge_types)
        log_msg(f'finished step 2')

        # Step 3: fix inconsistent edges
        log_msg(f'started step 3: fix inconsistent edges')
        fix_inconsistent_edges(adj_matrix_dp, heuristic)
        log_msg(f'finished step 3')
        
        # Step 4: fix zero degree nodes
        log_msg(f'started step 4: fix zero degree nodes')
        fix_zero_degree_nodes(adj_matrix_dp)
        log_msg(f'finished step 4')
        
        graph_dp = Graph(graph.name, adj_matrix_dp, graph.edge_types)

        log_msg(f'finished approach_1 execution')

        return graph_dp
