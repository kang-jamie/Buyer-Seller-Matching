# Buyer-Seller-Matching
## Intro:
I study online and batch matching algorithms for two-sided markets. Comparisons are made via simulations where I replicate agents with stochastic arrival/departure times and valuations. Work in progress.

## Files:
* traders.py: Simulates agents who arrive to the platform according to Poisson process, and departs stochastically. Their valuations are also drawn iid from some distribution.
* mechanism.py: Four dynamic matching algorithms. Online algorithms adapted from Blum et al. 2006 (https://www.cs.cmu.edu/~sandholm/online_clearing.jacm.pdf). 
* main.py: Main program to import and run the above files and plot using matplotlib and pandas.

## References:
Avrim Blum, Tuomas Sandholm, and Martin Zinkevich.  Online algorithms for marketclearing.Journal of the ACM (JACM), 53(5):845–879, 2006.
