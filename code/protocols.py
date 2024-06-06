from utils import *

from abc import ABC
from enum import Enum
import numpy as np


class ProtocolInit(Enum):
    OUE = 'oue'
    RR = 'rr'


class Protocol(ABC):
    def __init__(self, n:np.int_, d:np.int_, epsilon:np.float_) -> None:
        self.n = n
        self.d = d
        self.epsilon = epsilon
        self.p = self._get_p()
        self.q = self._get_q()
        self.counts_t = np.zeros(d)
        self.frequencies_t = np.zeros(d)


    def _get_p(self) -> np.float_:
        pass
    

    def _get_q(self) -> np.float_:
        pass


    def _update_counts_true(self, bit_vector:np.ndarray) -> None:
        self.counts_t += bit_vector


    def perturb(self, bit_vector:np.ndarray) -> np.ndarray:
        pass


    def update_frequencies_true(self) -> np.ndarray:
        self.frequencies_t = self.counts_t / np.sum(self.counts_t)


# https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao
class OUE(Protocol):
    def __init__(self, n:np.int_, d:np.int_, epsilon:np.float_) -> None:
        super(OUE, self).__init__(n, d, epsilon)

        self.n = n
        self.d = d
        self.epsilon = epsilon
        self.p = self._get_p()
        self.q = self._get_q()
        self.counts_t = np.zeros(d)
        self.frequencies_t = np.zeros(d)


    def _get_p(self) -> np.float_:
        return 1 / 2
    

    def _get_q(self) -> np.float_:
        return 1 / (np.exp(self.epsilon) + 1)
    

    def perturb(self, bit_vector:np.ndarray) -> np.ndarray:
        self._update_counts_true(bit_vector)

        bit_vector_copy = bit_vector.copy()

        prob_11 = self.p
        prob_10 = self.q
        prob_01 = 1 - self.p
        prob_00 = 1 - self.q

        val1_weights = [prob_11, prob_01]
        val0_weights = [prob_10, prob_00]

        population = [1, 0]

        val1_idx = np.where(bit_vector_copy == 1)[0]
        val0_idx = np.where(bit_vector_copy == 0)[0]
            
        val1_toss = np.random.choice(a=population, size=len(val1_idx), p=val1_weights)
        val0_toss = np.random.choice(a=population, size=len(val0_idx), p=val0_weights)

        bit_vector_copy[val1_idx] = val1_toss
        bit_vector_copy[val0_idx] = val0_toss
        
        return bit_vector_copy


# https://dl.acm.org/doi/10.1145/3133956.3134086
class RR(Protocol):
    def __init__(self, n:np.int_, d:np.int_, epsilon:np.float_) -> None:
        super(RR, self).__init__(n, d, epsilon)

        self.n = n
        self.d = d
        self.epsilon = epsilon
        self.p = self._get_p()
        self.q = self._get_q()
        self.counts_t = np.zeros(d)
        self.frequencies_t = np.zeros(d)


    def _get_p(self) -> np.float_:
        return np.exp(self.epsilon) / (np.exp(self.epsilon) + 1)
    

    def _get_q(self) -> np.float_:
        return 1 - self.p
    

    def perturb(self, bit_vector:np.ndarray) -> np.ndarray:
        self._update_counts_true(bit_vector)

        bit_vector_copy = bit_vector.copy()

        prob_11 = self.p
        prob_10 = self.q
        prob_01 = self.q
        prob_00 = self.p

        val1_weights = [prob_11, prob_01]
        val0_weights = [prob_10, prob_00]

        population = [1, 0]

        val1_idx = np.where(bit_vector_copy == 1)[0]
        val0_idx = np.where(bit_vector_copy == 0)[0]
            
        val1_toss = np.random.choice(a=population, size=len(val1_idx), p=val1_weights)
        val0_toss = np.random.choice(a=population, size=len(val0_idx), p=val0_weights)

        bit_vector_copy[val1_idx] = val1_toss
        bit_vector_copy[val0_idx] = val0_toss
        
        return bit_vector_copy


def init_protocol(protocol_init:ProtocolInit, *args) -> Protocol:
    if protocol_init == ProtocolInit.OUE:
        assert len(args) == 3

        return OUE(args[0], args[1], args[2])
    
    elif protocol_init == ProtocolInit.RR:
        assert len(args) == 3

        return RR(args[0], args[1], args[2])
    
    else:
        raise Exception
