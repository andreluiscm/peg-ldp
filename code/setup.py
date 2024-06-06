from databases import *


#####################
##    S E T U P    ##
#####################
CONST_DATABASES = [DBLP(), Netscience(), PierreAuger(), YeastLandscape()]
CONST_PART_CLUS_PAIRS = {
    DBLP().database_folder: (41, 34),
    Netscience().database_folder: (14, 24),
    PierreAuger().database_folder: (1, 8),
    YeastLandscape().database_folder: (4, 16)
}
CONST_EPSILONS = [0.1, 0.5, 1.0]
CONST_N_EXECUTIONS = 10
CONST_DUMP_GRAPH = True
#####################


class Setup():
    def __init__(self) -> None:
        self.databases = CONST_DATABASES
        self.part_clus_pairs = CONST_PART_CLUS_PAIRS
        self.epsilons = CONST_EPSILONS
        self.n_executions = CONST_N_EXECUTIONS
        self.dump_graph = CONST_DUMP_GRAPH

    def print_info(self) -> None:
        print(f'\n-- SETUP INFO --')
        print(f'.')
        print(f'databases = {self.databases}')
        print(f'epsilons = {self.epsilons}')
        print(f'# executions = {self.n_executions}')
        print(f'dump graph = {self.dump_graph}')
        print(f'.')
