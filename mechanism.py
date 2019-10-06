### mechanism.py
# This class uses four different mechanisms to allocate items
# in the double auction for given bids and asks
# 
#


## Import modules
from mosek.fusion import *
import numpy as np
import math
import networkx as nx


class Mechanism(object):
    def __init__(self, B, S, B_arrival, S_arrival, B_depart, S_depart, **kwargs):
        """
        Initializes the Mechanism class

        """
        self.B = B
        self.S = S
        self.B_arrival = B_arrival
        self.S_arrival = S_arrival
        self.B_depart = B_depart
        self.S_depart = S_depart
    
    def __str__(self):
        pass

    def run_mechanism(self):
        pass
        
    def payment(self):
        pass 

class Mech_Batch_Welfare(Mechanism):
    def __init__(self, d, **kwargs):
        Mechanism.__init__(self, **kwargs)
        self.B_M = np.array([], dtype=int)
        self.S_M = np.array([], dtype=int)
        self.M_num = 0
        self.d = d

    def __str__(self):
        return "Mech_Batch_Welfare"    

    def run_mechanism(self):
        k_max = max(int(self.B_depart[-1]//self.d), int(self.S_depart[-1]//self.d))
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
        # print("Matches:", list(zip(self.B[self.B_M], self.S[self.S_M])))
        # print("Welfare:", sum(self.B[self.B_M]) + sum(self.S[self.S_M]))
        # print("M_num:", self.M_num)
        return self.B[self.B_M], self.S[self.S_M]

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
        pass

class Mech_Online_Welfare(Mechanism):
    def __init__(self, p_min, p_max, alpha, r, **kwargs):
        Mechanism.__init__(self, **kwargs)
        self.p_min = p_min
        self.p_max = p_max
        self.alpha = alpha
        self.r = r

        self.B_M = np.array([], dtype=int)
        self.S_M = np.array([], dtype=int)
        self.M_num = 0

    def __str__(self):
        return "Mech_Random_Welfare"

    def run_mechanism(self):
        theta = self.simulate_thresh()
        # print("theta:", theta)

        B_alive_idx = np.array([],dtype=int) #Store indices of alive buy bids
        S_alive_idx = np.array([],dtype=int) #Store indices of alive sell bids
        B_current = 0
        S_current = 0

        counter = 0
        while B_current < len(self.B) or S_current < len(self.S):
            counter = counter + 1
            # Determine next event
            if B_current >= len(self.B):
                next_arrival = self.S_arrival[S_current]
                next_type = 'S'
                # print("case1")
            elif S_current >= len(self.S):
                next_arrival = self.B_arrival[B_current]
                next_type = 'B'
                # print("case2")
            elif self.B_arrival[B_current] < self.S_arrival[S_current]:
                next_arrival = self.B_arrival[B_current]
                next_type = 'B'
                # print("case3")
            else: 
                next_arrival = self.S_arrival[S_current]
                next_type = 'S'
                # print("case4")
            # print("next_type:", next_type)
            # print("B_current:", B_current)
            # print("S_current:", S_current)
            # Remove departing bids
            if B_alive_idx.size != 0: 
                B_alive_idx = B_alive_idx[np.where(self.B_depart[B_alive_idx] > next_arrival)]
            if S_alive_idx.size != 0: 
                S_alive_idx = S_alive_idx[np.where(self.S_depart[S_alive_idx] > next_arrival)]
            # print("Time:", next_arrival)
            # print("Counter: ", counter)
            # print("B_alive:", self.B[B_alive_idx])
            # print("S_alive:", self.S[S_alive_idx])

            # Next arriving bid (buying)
            if next_type == 'B':
                # print("Arriving b:", self.B[B_current])
                # Discard it
                if self.B[B_current] < theta:
                # print("buy bid too low!")
                    pass
                # Match it with random sell bid
                elif S_alive_idx.size != 0:
                    self.B_M = np.hstack((self.B_M, B_current))
                    S_M_new = np.random.choice(S_alive_idx)
                    self.S_M = np.hstack((self.S_M, S_M_new))
                    S_alive_idx = np.setdiff1d(S_alive_idx, S_M_new)
                    # print("matched!!")
                    self.M_num = self.M_num + 1
                # Don't match it
                else:
                    B_alive_idx = np.append(B_alive_idx,B_current)      
                    # print("added!!")
                B_current = B_current + 1
            else: 
                # print("Arriving s:", self.S[S_current])
                # Discard it
                if self.S[S_current] > theta:
                    # print("sell bid too high!")
                    pass
                # Match it with random buy bid
                elif B_alive_idx.size != 0:
                    self.S_M = np.hstack((self.S_M, S_current))
                    B_M_new = np.random.choice(B_alive_idx)
                    self.B_M = np.hstack((self.B_M, B_M_new))
                    B_alive_idx = np.setdiff1d(B_alive_idx, B_M_new)
                    # print("matched!!")
                    self.M_num = self.M_num + 1
                # Don't match it
                else:
                    S_alive_idx = np.append(S_alive_idx,S_current)
                    # print("added")
                S_current = S_current + 1
        # print("Matches:", list(zip(self.B[self.B_M], self.S[self.S_M])))
        # print("Welfare:", sum(self.B[self.B_M]) + sum(self.S[self.S_M]))
        # print("M_num:", self.M_num)
        # print("==================")
        return self.B[self.B_M], self.S[self.S_M]

    def simulate_thresh(self):
        u = np.random.random()
        theta = self.p_min + (self.r-1)*self.p_min*np.exp(u*self.r*self.alpha)
        return theta


class Mech_Batch_Liquidity(Mechanism):
    def __init__(self, d, **kwargs):
        Mechanism.__init__(self, **kwargs)
        self.B_M = np.array([], dtype=int)
        self.S_M = np.array([], dtype=int)
        self.M_num = 0
        self.d = d

    def __str__(self):
        return "Mech_Batch_Liquidity"

    def run_mechanism(self):
        k_max = max(int(self.B_depart[-1]//self.d), int(self.S_depart[-1]//self.d))
        B_alive_idx = np.array([],dtype=int) #Store indices of alive buy bids
        S_alive_idx = np.array([],dtype=int) #Store indices of alive sell bids
        B_current = 0
        S_current = 0
        G = nx.DiGraph()
        for k in range(k_max):
            start = k*self.d 
            end = (k+1) * self.d
            # Remove departing bids
            if B_alive_idx.size != 0:
                B_depart_idx = B_alive_idx[np.where(self.B_depart[B_alive_idx] <= end)]
                B_alive_idx = np.setdiff1d(B_alive_idx,B_depart_idx)
                for b in B_depart_idx:
                    if (-b-1) in G:
                        G.remove_node(-b-1)
                    
            if S_alive_idx.size != 0:
                S_depart_idx = S_alive_idx[np.where(self.S_depart[S_alive_idx] <= end)]
                S_alive_idx = np.setdiff1d(S_alive_idx,S_depart_idx)
                for s in S_depart_idx:
                    if (s+1) in G:
                        G.remove_node(s+1)

            # Add arriving bids
            while B_current < len(self.B) and self.B_arrival[B_current] <= end:
                B_alive_idx = np.append(B_alive_idx,B_current)
                S_candidates = S_alive_idx[np.where(self.S[S_alive_idx] <= self.B[B_current])]
                for s in S_candidates:
                    G.add_edge(s+1, -B_current-1)         
                B_current = B_current + 1

            while S_current < len(self.S) and self.S_arrival[S_current] <= end:
                S_alive_idx = np.append(S_alive_idx,S_current)
                B_candidates = B_alive_idx[np.where(self.B[B_alive_idx] >= self.S[S_current])]
                for b in B_candidates:
                    G.add_edge(S_current+1, -b-1)
                S_current = S_current + 1


            # Clear matching
            M_new = nx.maximal_matching(G)
            M_num_new = len(M_new)
            if np.array(list(M_new)).size != 0:
                S_M_new = np.array(list(M_new))[:,0]-1
                B_M_new = -np.array(list(M_new))[:,1]-1

                self.B_M = np.hstack((self.B_M, B_M_new))
                self.S_M = np.hstack((self.S_M, S_M_new))
                self.M_num = self.M_num + M_num_new
                B_alive_idx = np.setdiff1d(B_alive_idx, B_M_new)
                G.remove_nodes_from(-B_M_new-1)
                S_alive_idx = np.setdiff1d(S_alive_idx, S_M_new)
                G.remove_nodes_from(S_M_new-1)
        return self.B[self.B_M], self.S[self.S_M]

    # def matching(self, B_alive_idx, S_alive_idx)
    #     B_alive = self.B[B_alive_idx]
    #     S_alive = self.S[S_alive_idx]




class Mech_Greedy_Liquidity(Mechanism):
    def __init__(self, **kwargs):
        Mechanism.__init__(self, **kwargs)

        self.B_M = np.array([], dtype=int)
        self.S_M = np.array([], dtype=int)
        self.M_num = 0

    def __str__(self):
        return "Mech_Random_Welfare"

    def run_mechanism(self):
        B_alive_idx = np.array([],dtype=int) #Store indices of alive buy bids
        S_alive_idx = np.array([],dtype=int) #Store indices of alive sell bids
        B_current = 0
        S_current = 0

        counter = 0
        while B_current < len(self.B) or S_current < len(self.S):
            counter = counter + 1
            # Determine next event
            if B_current >= len(self.B):
                next_arrival = self.S_arrival[S_current]
                next_type = 'S'
                # print("case1")
            elif S_current >= len(self.S):
                next_arrival = self.B_arrival[B_current]
                next_type = 'B'
                # print("case2")
            elif self.B_arrival[B_current] < self.S_arrival[S_current]:
                next_arrival = self.B_arrival[B_current]
                next_type = 'B'
                # print("case3")
            else: 
                next_arrival = self.S_arrival[S_current]
                next_type = 'S'
                # print("case4")

            # Remove departing bids
            if B_alive_idx.size != 0: 
                B_alive_idx = B_alive_idx[np.where(self.B_depart[B_alive_idx] > next_arrival)]

            if S_alive_idx.size != 0: 
                S_alive_idx = S_alive_idx[np.where(self.S_depart[S_alive_idx] > next_arrival)]

            # Next arriving bid (buying)
            if next_type == 'B':
                S_candidates = S_alive_idx[np.where(self.S[S_alive_idx] <= self.B[B_current])]

                # Match it with random sell bid
                if S_candidates.size != 0:
                    self.B_M = np.hstack((self.B_M, B_current))
                    S_M_new = np.random.choice(S_candidates)
                    self.S_M = np.hstack((self.S_M, S_M_new))
                    S_alive_idx = np.setdiff1d(S_alive_idx, S_M_new)
                    self.M_num = self.M_num + 1
                # Don't match it
                else:
                    B_alive_idx = np.append(B_alive_idx,B_current)      
                    # print("added!!")
                B_current = B_current + 1

            else: 
                B_candidates = B_alive_idx[np.where(self.B[B_alive_idx] >= self.S[S_current])]
                # Match it with random buy bid
                if B_candidates.size != 0:
                    self.S_M = np.hstack((self.S_M, S_current))
                    B_M_new = np.random.choice(B_candidates)
                    self.B_M = np.hstack((self.B_M, B_M_new))
                    B_alive_idx = np.setdiff1d(B_alive_idx, B_M_new)
                    self.M_num = self.M_num + 1
                # Don't match it
                else:
                    S_alive_idx = np.append(S_alive_idx,S_current)
                    # print("added")
                S_current = S_current + 1
        # print("Matches:", list(zip(self.B[self.B_M], self.S[self.S_M])))
        # print("Welfare:", sum(self.B[self.B_M]) + sum(self.S[self.S_M]))
        # print("M_num:", self.M_num)
        # print("==================")
        return self.B[self.B_M], self.S[self.S_M]
