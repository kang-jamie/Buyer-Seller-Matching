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

		return 	B_M_new, S_M_new, M_num_new

	def payment(self):
		#TODO

class Mech_BMM(Mechanism):

	def bmmRule(self):
		"""
		BMM rule (BMM)
		This function calls self.splitBatch to split bids and asks into batches
		and then calls self.bmmSingleBatch for each batch.
		Any unmatched bids/asks from previous batches are rolled over.
		Returns the total number of matches.
		"""
		self.splitBatch()
		counter = 0
		unmatched_bids = []
		unmatched_asks = []

		for n in range(self.num_batch):
			this_bids = self.bid_batch[n]
			this_bids.extend(unmatched_bids)
			this_asks = self.ask_batch[n]
			this_asks.extend(unmatched_asks)
			this_counter, unmatched_bid_idx, unmatched_ask_idx, matched_bid_idx, matched_ask_idx \
				= self.bmmSingleBatch(this_bids, this_asks)
			counter += this_counter

			unmatched_bids = [this_bids[idx] for idx in unmatched_bid_idx]
			unmatched_asks = [this_asks[idx] for idx in unmatched_ask_idx]
		return counter

	def bmmWeightedRule(self):
		"""
		BMM rule (BMM)
		This function calls self.splitBatch to split bids and asks into batches
		and then calls self.bmmSingleBatch for each batch.
		Any unmatched bids/asks from previous batches are rolled over.
		Returns the total number of matches.
		"""
		self.splitBatch()
		counter = 0
		unmatched_bids = []
		unmatched_asks = []

		for n in range(self.num_batch):
			this_bids = self.bid_batch[n]
			this_bids.extend(unmatched_bids)
			this_asks = self.ask_batch[n]
			this_asks.extend(unmatched_asks)
			this_counter, unmatched_bid_idx, unmatched_ask_idx, matched_bid_idx, matched_ask_idx \
				= self.twoPhaseSingleBatch(this_bids, this_asks)
			counter += this_counter

			unmatched_bids = [this_bids[idx] for idx in unmatched_bid_idx]
			unmatched_asks = [this_asks[idx] for idx in unmatched_ask_idx]
		return counter

	def twoPhaseRule(self):
		self.splitBatch()
		counter = 0
		unmatched_bids = []
		unmatched_asks = []

		for n in range(self.num_batch):
			this_bids = self.bid_batch[n]
			this_bids.extend(unmatched_bids)
			this_asks = self.ask_batch[n]
			this_asks.extend(unmatched_asks)
			this_counter, unmatched_bid_idx, unmatched_ask_idx, matched_bid_idx, matched_ask_idx \
				= self.bmmWeightedSingleBatch(this_bids, this_asks)
			counter += this_counter

			unmatched_bids = [this_bids[idx] for idx in unmatched_bid_idx]
			unmatched_asks = [this_asks[idx] for idx in unmatched_ask_idx]
		return counter

	def splitBatch(self):
		"""
		Split bids and asks into batches
		"""
		self.num_batch = math.ceil(self.T/self.period)
		self.bid_batch = [[] for _ in range(self.num_batch)]
		self.ask_batch = [[] for _ in range(self.num_batch)]

		for i, time in enumerate(self.bid_times):
			n = math.floor(time/self.period)
			self.bid_batch[n].append(self.total_bids[i])
		for i, time in enumerate(self.ask_times):
			n = math.floor(time/self.period)
			self.ask_batch[n].append(self.total_asks[i])


	def streamprinter(text):
		"""
		Sys function for MOSEK solver
		"""
		sys.stdout.write(text)
		sys.stdout.flush()

	def twoPhaseSingleBatch(self, bids, asks):
		"""
		BMM for each batch
		This function uses MOSEK solver to solve sparse bipartite graph matching problem

		* Returns: 
		num_match: number of matches from current batch (plus rolled over bids/asks)
		unmatched_bids, unmatched_asks
		matched_bids, matched_asks
		"""

		## PHASE I
		_, good_bids, good_asks = self.onePriceRule()

		## PHASE II
		bid_num = len(bids)
		ask_num = len(asks)

		# Bid or Ask list is empty (extreme case, usually doesnt happen)
		if bid_num == 0 or ask_num == 0:
			# print("empty bid and/or ask list")
			return 0, [], [], [], []

		else:
			# Preprocess decision variables to reduce dimensionality
			feasible_pairs = []
			pair_index = 0
			feasible_pairs_with_bid_i = [[] for _ in range(bid_num)]
			feasible_pairs_with_ask_j = [[] for _ in range(ask_num)]
			for i in range(bid_num):
				for j in range(ask_num):
					if bids[i] >= asks[j]:
						feasible_pairs.append((i,j))
						feasible_pairs_with_bid_i[i].append(pair_index)
						feasible_pairs_with_ask_j[j].append(pair_index)
						pair_index += 1

			num_feasible = pair_index
			feasible_bids = [i for (i, v) in enumerate(feasible_pairs_with_bid_i) if v]
			feasible_asks = [j for (j, v) in enumerate(feasible_pairs_with_ask_j) if v]


			# Optimize using MOSEK
			with Model("BatchMatchingModel") as M:
				x = M.variable("x", num_feasible, Domain.greaterThan(0.0))

				# Each (feasible) bid can be matched upto once
				for i in feasible_bids:
					j_list = feasible_pairs_with_bid_i[i]
					bid_list = x.pick(j_list)
					if i in good_bids:
						M.constraint(Expr.sum(bid_list), Domain.equalsTo(1.0))
					else:
						M.constraint(Expr.sum(bid_list), Domain.lessThan(1.0))
				# Each (feasible) ask can be matched upto once
				for j in feasible_asks:
					i_list = feasible_pairs_with_ask_j[j]
					ask_list = x.pick(i_list)
					if j in good_asks:
						M.constraint(Expr.sum(ask_list), Domain.equalsTo(1.0))
					else:
						M.constraint(Expr.sum(ask_list), Domain.lessThan(1.0))
				
				M.constraint(x, Domain.lessThan(1.0)) #maybe this can be relaxed?
				M.objective(ObjectiveSense.Maximize, Expr.sum(x))
				M.solve()
				X = x.level()
				X = np.round(X,decimals=1)
				num_match = M.primalObjValue()
				num_match = int(num_match)
				match_index = np.nonzero(X)[0]
				matched_ij = [feasible_pairs[m] for m in match_index]
				matched_bids = [m[0] for m in matched_ij]
				matched_asks = [m[1] for m in matched_ij]
				unmatched_bids = [i for i in range(bid_num) if i not in matched_bids]
				unmatched_asks = [j for j in range(ask_num) if j not in matched_asks]


		# Return number of matches and lists of unmatched and matched bids and asks
		return num_match, unmatched_bids, unmatched_asks, matched_bids, matched_asks

	def bmmSingleBatch(self, bids, asks):
		"""
		BMM for each batch
		This function uses MOSEK solver to solve sparse bipartite graph matching problem

		* Returns: 
		num_match: number of matches from current batch (plus rolled over bids/asks)
		unmatched_bids, unmatched_asks
		matched_bids, matched_asks
		"""
		bid_num = len(bids)
		ask_num = len(asks)

		# Bid or Ask list is empty (extreme case, usually doesnt happen)
		if bid_num == 0 or ask_num == 0:
			# print("empty bid and/or ask list")
			return 0, [], [], [], []

		else:
			# Preprocess decision variables to reduce dimensionality
			feasible_pairs = []
			pair_index = 0
			feasible_pairs_with_bid_i = [[] for _ in range(bid_num)]
			feasible_pairs_with_ask_j = [[] for _ in range(ask_num)]
			for i in range(bid_num):
				for j in range(ask_num):
					if bids[i] >= asks[j]:
						feasible_pairs.append((i,j))
						feasible_pairs_with_bid_i[i].append(pair_index)
						feasible_pairs_with_ask_j[j].append(pair_index)
						pair_index += 1

			num_feasible = pair_index
			feasible_bids = [i for (i, v) in enumerate(feasible_pairs_with_bid_i) if v]
			feasible_asks = [j for (j, v) in enumerate(feasible_pairs_with_ask_j) if v]


			# Optimize using MOSEK
			with Model("BatchMatchingModel") as M:
				x = M.variable("x", num_feasible, Domain.greaterThan(0.0))

				# Each (feasible) bid can be matched upto once
				for i in feasible_bids:
					j_list = feasible_pairs_with_bid_i[i]
					bid_list = x.pick(j_list)
					M.constraint(Expr.sum(bid_list), Domain.lessThan(1.0))
				# Each (feasible) ask can be matched upto once
				for j in feasible_asks:
					i_list = feasible_pairs_with_ask_j[j]
					ask_list = x.pick(i_list)
					M.constraint(Expr.sum(ask_list), Domain.lessThan(1.0))
				
				M.constraint(x, Domain.lessThan(1.0)) #maybe this can be relaxed?
				M.objective(ObjectiveSense.Maximize, Expr.sum(x))
				M.solve()
				X = x.level()
				X = np.round(X,decimals=1)
				num_match = M.primalObjValue()
				num_match = int(num_match)
				match_index = np.nonzero(X)[0]
				matched_ij = [feasible_pairs[m] for m in match_index]
				matched_bids = [m[0] for m in matched_ij]
				matched_asks = [m[1] for m in matched_ij]
				unmatched_bids = [i for i in range(bid_num) if i not in matched_bids]
				unmatched_asks = [j for j in range(ask_num) if j not in matched_asks]


		# Return number of matches and lists of unmatched and matched bids and asks
		return num_match, unmatched_bids, unmatched_asks, matched_bids, matched_asks


	def bmmWeightedSingleBatch(self, bids, asks):
		"""
		BMM for each batch
		This function uses MOSEK solver to solve sparse bipartite graph matching problem

		* Returns: 
		num_match: number of matches from current batch (plus rolled over bids/asks)
		unmatched_bids, unmatched_asks
		matched_bids, matched_asks
		"""
		bid_num = len(bids)
		ask_num = len(asks)

		# Bid or Ask list is empty (extreme case, usually doesnt happen)
		if bid_num == 0 or ask_num == 0:
			# print("empty bid and/or ask list")
			return 0, [], [], [], []

		else:
			# Preprocess decision variables to reduce dimensionality
			feasible_pairs = []
			pair_value = []
			pair_index = 0
			feasible_pairs_with_bid_i = [[] for _ in range(bid_num)]
			feasible_pairs_with_ask_j = [[] for _ in range(ask_num)]
			for i in range(bid_num):
				for j in range(ask_num):
					if bids[i] >= asks[j]:
						feasible_pairs.append((i,j))
						feasible_pairs_with_bid_i[i].append(pair_index)
						feasible_pairs_with_ask_j[j].append(pair_index)
						pair_value.append(bids[i] - asks[j])
						pair_index += 1
						#TODO: store bid-ask spread separately to put in optimization objective

			num_feasible = pair_index
			feasible_bids = [i for (i, v) in enumerate(feasible_pairs_with_bid_i) if v]
			feasible_asks = [j for (j, v) in enumerate(feasible_pairs_with_ask_j) if v]


			# Optimize using MOSEK
			with Model("BatchMatchingModel") as M:
				x = M.variable("x", num_feasible, Domain.greaterThan(0.0))
				# Each (feasible) bid can be matched upto once
				for i in feasible_bids:
					j_list = feasible_pairs_with_bid_i[i]
					bid_list = x.pick(j_list)
					M.constraint(Expr.sum(bid_list), Domain.lessThan(1.0))
				# Each (feasible) ask can be matched upto once
				for j in feasible_asks:
					i_list = feasible_pairs_with_ask_j[j]
					ask_list = x.pick(i_list)
					M.constraint(Expr.sum(ask_list), Domain.lessThan(1.0))
				
				M.constraint(x, Domain.lessThan(1.0)) #maybe this can be relaxed?
				# M.objective(ObjectiveSense.Maximize, Expr.sum(x))
				M.objective(ObjectiveSense.Maximize, Expr.dot(x, pair_value))
				M.solve()
				X = x.level()
				X = np.round(X,decimals=1)
				# num_match = M.primalObjValue()
				# num_match = int(num_match)
				num_match = np.count_nonzero(X)
				match_index = np.nonzero(X)[0]
				matched_ij = [feasible_pairs[m] for m in match_index]
				matched_bids = [m[0] for m in matched_ij]
				matched_asks = [m[1] for m in matched_ij]
				unmatched_bids = [i for i in range(bid_num) if i not in matched_bids]
				unmatched_asks = [j for j in range(ask_num) if j not in matched_asks]


		# Return number of matches and lists of unmatched and matched bids and asks
		return num_match, unmatched_bids, unmatched_asks, matched_bids, matched_asks

	def bmmWeightedSingleBatch(self, bids, asks):
		"""
		BMM for each batch
		This function uses MOSEK solver to solve sparse bipartite graph matching problem

		* Returns: 
		num_match: number of matches from current batch (plus rolled over bids/asks)
		unmatched_bids, unmatched_asks
		matched_bids, matched_asks
		"""
		bid_num = len(bids)
		ask_num = len(asks)

		# Bid or Ask list is empty (extreme case, usually doesnt happen)
		if bid_num == 0 or ask_num == 0:
			# print("empty bid and/or ask list")
			return 0, [], [], [], []

		else:
			# Preprocess decision variables to reduce dimensionality
			feasible_pairs = []
			pair_value = []
			pair_index = 0
			feasible_pairs_with_bid_i = [[] for _ in range(bid_num)]
			feasible_pairs_with_ask_j = [[] for _ in range(ask_num)]
			for i in range(bid_num):
				for j in range(ask_num):
					if bids[i] >= asks[j]:
						feasible_pairs.append((i,j))
						feasible_pairs_with_bid_i[i].append(pair_index)
						feasible_pairs_with_ask_j[j].append(pair_index)
						pair_value.append(bids[i] - asks[j])
						pair_index += 1
						#TODO: store bid-ask spread separately to put in optimization objective

			num_feasible = pair_index
			feasible_bids = [i for (i, v) in enumerate(feasible_pairs_with_bid_i) if v]
			feasible_asks = [j for (j, v) in enumerate(feasible_pairs_with_ask_j) if v]


			# Optimize using MOSEK
			with Model("BatchMatchingModel") as M:
				x = M.variable("x", num_feasible, Domain.greaterThan(0.0))
				# Each (feasible) bid can be matched upto once
				for i in feasible_bids:
					j_list = feasible_pairs_with_bid_i[i]
					bid_list = x.pick(j_list)
					M.constraint(Expr.sum(bid_list), Domain.lessThan(1.0))
				# Each (feasible) ask can be matched upto once
				for j in feasible_asks:
					i_list = feasible_pairs_with_ask_j[j]
					ask_list = x.pick(i_list)
					M.constraint(Expr.sum(ask_list), Domain.lessThan(1.0))
				
				M.constraint(x, Domain.lessThan(1.0)) #maybe this can be relaxed?
				# M.objective(ObjectiveSense.Maximize, Expr.sum(x))
				M.objective(ObjectiveSense.Maximize, Expr.dot(x, pair_value))
				M.solve()
				X = x.level()
				X = np.round(X,decimals=1)
				# num_match = M.primalObjValue()
				# num_match = int(num_match)
				num_match = np.count_nonzero(X)
				match_index = np.nonzero(X)[0]
				matched_ij = [feasible_pairs[m] for m in match_index]
				matched_bids = [m[0] for m in matched_ij]
				matched_asks = [m[1] for m in matched_ij]
				unmatched_bids = [i for i in range(bid_num) if i not in matched_bids]
				unmatched_asks = [j for j in range(ask_num) if j not in matched_asks]


		# Return number of matches and lists of unmatched and matched bids and asks
		return num_match, unmatched_bids, unmatched_asks, matched_bids, matched_asks
