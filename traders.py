### BLUE OCEAN PROJECT: simulateTraders.py
# This class simulates random trader arrivals and their valuations
#
# Author: Jamie Kang (jamiekang@stanford.edu)
# Institute: MS&E Dept., Stanford University
# Last Edited: 08/2019

## Import modules
import numpy as np


class Trader(object):
    def __init__(self, lam, T, price_low, price_high):
        """
        Initializes the simulator

        Inputs:
        lam: arrival intensity (i.e. market liquidity)
        T: total time horizon
        price_low/price_high: lower/upper bound for valuation
        """
        self.lam = lam
        self.T = T
        self.price_low = price_low
        self.price_high = price_high

        # Store arrivals in self.arrival
        self.arrival = []
        # Count number of arrivaㅣ
        self.total_num = 0

    def __str__(self):
        pass

    def simulate(self):
        """
        Simulate traders and their valuations by calling simulate_arrival and simulate_value 
        Returns the valuations, arrival times, and the total number of arrivals
        """
        self.simulate_arrival()
        self.simulate_value()
        return self.values, self.arrival, self.total_num

    def simulate_arrival(self):
        """
        Simulate arrival times as Poisson process by generating interarrival times
        as exponential distribution random variables
        """
        self.current_time = 0
        while True:
            inter_arrival = np.random.exponential(scale = 1/self.lam) #or 1/lam?
            if self.current_time + inter_arrival >= self.T:
                break
            self.current_time += inter_arrival
            self.arrival.append(self.current_time)

            # Count the number of arrivals
            self.total_num += 1

    def simulate_value(self):
        """
        Simulate each trader's valuation by generating uniform distribution random variables
        """
        self.values = np.random.uniform(low=self.price_low, high=self.price_high, size=self.total_num)


    def simulate_num(self):
        self.values = np.random.randint