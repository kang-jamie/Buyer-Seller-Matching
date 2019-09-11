### traders.py
# This class simulates random trader arrivals and their valuations
#
#

## Import modules
import numpy as np


class Trader(object):
    def __init__(self, arrival_rate, depart_rate, d, T, price_low, price_high):
        """
        Initializes the simulator

        Inputs:
        arrival_rate: arrival intensity (i.e. market liquidity)
        T: total time horizon
        price_low/price_high: lower/upper bound for valuation
        d: patience delta
        """
        self.arrival_rate = arrival_rate
        self.depart_rate = depart_rate
        self.d = d
        self.T = T
        self.price_low = price_low
        self.price_high = price_high

        # Store arrivals in self.arrival
        # self.arrival = np.array([])
        # self.depart = np.array([])
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
        self.simulate_departure()
        return self.values, self.arrival, self.depart, self.total_num

    def simulate_arrival(self):
        """
        Simulate arrival times as Poisson process by generating interarrival times
        as exponential distribution random variables
        """
        # self.arrival = np.array([])
        self.arrival = []
        self.current_time = 0
        while True:
            inter_arrival = np.random.exponential(scale = 1/self.arrival_rate) #or 1/arrival_rate?
            if self.current_time + inter_arrival >= self.T:
                break
            self.current_time += inter_arrival
            self.arrival.append(self.current_time)
            # np.concatenate([self.arrival, self.current_time])
            # Count the number of arrivals
            self.total_num += 1
        self.arrival = np.asarray(self.arrival)

    def simulate_departure(self):
        self.depart = self.arrival + self.d + np.random.exponential(scale = 1/self.depart_rate, size=self.total_num)
        # self.depart = self.arrival + self.d + np.random.uniform(low = 0, high = self.depart_rate*2, size=self.total_num)

    def simulate_value(self):
        """
        Simulate each trader's valuation by generating uniform distribution random variables
        """
        self.values = np.random.uniform(low=self.price_low, high=self.price_high, size=self.total_num)


    def simulate_num(self):
        self.values = np.random.randint