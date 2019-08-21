### mechanism.py
# This class uses two different methods (OP and BMM) to allocate items
# in the double auction for given bids and asks
# 
#
# This class uses MOSEK solver. Need to install mosek.fusion
# (https://docs.mosek.com/8.1/pythonfusion/install-interface.html)
#


## Import modules
from mosek.fusion import *
import numpy as np
import math

class Mechanism(object):
    def __init__(self, B, S, B_arrival, S_arrival, B_depart, S_depart, d, T, **kwargs):
         """
        Initializes the Allcation class

        *Inputs:
        total_bids, total_asks
        bid_times, ask_times
        period: batch length
        T: total time horizon
        """
        self.total_bids = total_bids
        self.total_asks = total_asks
        self.bid_times = bid_times
        self.ask_times = ask_times
        self.d = d
        self.T = T
    
    def __str__(self):
        pass

    def run_mechanism(self):
        pass

    def matching(self):
        pass
        
    def payment(self):
        pass 

class Mech_Batch_Welfare(Mechanism):
    def __init__(self, **kwargs):
        Mechanism.__init__(self, **kwargs)
        self.B_M = np.array([])
        self.S_M = np.array([])

    def __str__(self):
        return "Mech_Batch_Welfare"    

    def run_mechanism(self):
        k_max = max(int(self.B_depart[-1]//self.d), int(self.S_depart[-1]//self.d))
        # print("k_max", k_max)

        B_alive_idx = np.array([],dtype=int) #Store indices of alive buy bids
        S_alive_idx = np.array([],dtype=int) #Store indices of alive sell bids
        B_current = 0
        S_current = 0
        for k in range(k_max):
            start = k*self.d 
            end = (k+1) * self.d

            # Remove departing bids
            if B_alive_idx.size != 0:
                B_alive_idx = B_alive_idx[np.where(self.B_depart[B_alive_idx] > end)] #Remove departing buy bids
            if S_alive_idx.size != 0:
                S_alive_idx = S_alive_idx[np.where(self.S_depart[S_alive_idx] > end)] #Remove departing sell bids

            # Add arriving bids
            while B_current < len(self.B) and self.B_arrival[B_current] <= end:
                B_alive_idx = np.append(B_alive_idx,B_current)
                B_current = B_current + 1
            while S_current < len(self.S) and self.S_arrival[S_current] <= end:
                S_alive_idx = np.append(S_alive_idx,S_current)
                S_current = S_current + 1


            # Clear matching
            B_M_new, S_M_new, M_num_new = self.matching(B_alive_idx, S_alive_idx)
            self.B_M = np.hstack((self.B_M, B_M_new))
            self.S_M = np.hstack((self.S_M, S_M_new))
            self.M_num = self.M_num + M_num_new
            B_alive_idx = np.setdiff1d(B_alive_idx, B_M_new)
            S_alive_idx = np.setdiff1d(S_alive_idx, S_M_new)


    def matching(self, B_alive_idx, S_alive_idx):
        B_alive = self.B[B_alive_idx]
        S_alive = self.S[S_alive_idx]


        B_sorted = sorted(B_alive, reverse=True) #decreasing order
        B_sorted_idx = np.argsort(-B_alive)
        S_sorted = sorted(S_alive, reverse=False) #increasing order  
        S_sorted_idx = np.argsort(S_alive)

        M_num_new = sum(b >= s for b,s in zip(B_sorted, S_sorted))

        temp_B = B_sorted_idx[range(M_num_new)]
        B_M_new = B_alive_idx[temp_B]
        temp_S = S_sorted_idx[range(M_num_new)]
        S_M_new = S_alive_idx[temp_S]

        return  B_M_new, S_M_new, M_num_new

    def payment(self):
        #TODO

