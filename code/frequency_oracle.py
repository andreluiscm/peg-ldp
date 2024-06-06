from utils import *

from abc import ABC
import numpy as np


class FrequencyOracle(ABC):
    def __init__(self, n:np.int_) -> None:
        self.responses = []
        self.n = n


    def aggregate(self) -> None:
        return
    

    def convert_responses_to_array(self) -> None:
        if type(self.responses) is not np.ndarray:
            self.responses = np.array(self.responses)


class FrequencyOracleProtocol(FrequencyOracle):
    def __init__(self, n:np.int_, d:np.int_, p:np.float_, q:np.float_) -> None:
        super(FrequencyOracleProtocol, self).__init__(n)

        self.d = d
        self.p = p
        self.q = q
        self.counts_p = np.zeros(d)
        self.counts_e = np.zeros(d)
        self.frequencies_p = np.zeros(d)
        self.frequencies_e = np.zeros(d)

    
    def aggregate(self, response:np.ndarray) -> None:
        self.responses.append(response)
    

    def estimate(self) -> None:
        self.convert_responses_to_array()
        self.counts_p = np.sum(self.responses, axis=0)

        counts_e = (self.counts_p - (self.q * self.n)) / (self.p - self.q)
        expected_sum = np.sum(counts_e)
        
        if expected_sum > 0:
            self.counts_e = post_process_counts_float(self.counts_p, expected_sum)
        else:
            self.counts_e = np.clip(counts_e, 0, np.max(counts_e))

        self.frequencies_p = self.counts_p / np.sum(self.counts_p)
        self.frequencies_e = self.counts_e / np.sum(self.counts_e)


class FrequencyOracleMechanism(FrequencyOracle):
    def __init__(self, n:np.int_) -> None:
        super(FrequencyOracleMechanism, self).__init__(n)


    def aggregate(self, responses:np.ndarray) -> None:
        self.responses = responses

    
    def post_process_responses(self, min_value:np.int_, max_value:np.int_) -> None:
        self.convert_responses_to_array()
        
        expected_sum = np.sum(self.responses)

        if expected_sum > 0:
            self.responses = post_process_counts_int(self.responses, expected_sum, min_value, max_value)
        else:
            self.responses = np.clip(self.responses, 0, max_value)
