import numpy as np
import matplotlib.pyplot as plt

import traders
import mechanism as mech

def MC_main(a_rate,d_rate,d,T):
    M_num_batch = 0
    W_batch = 0
    M_num_online = 0
    W_online = 0
    M_num_greedy = 0
    W_greedy = 0
    N = 100

    for i in range(N):
        buyer = traders.Trader(arrival_rate=a_rate, depart_rate=d_rate, d=d, T=T, price_low = price_low, price_high = price_high)
        B, B_arrival, B_depart, B_total = buyer.simulate()

        ## Simulate asks
        seller = traders.Trader(arrival_rate=a_rate, depart_rate=d_rate, d=d, T=T, price_low = price_low, price_high = price_high)
        S, S_arrival, S_depart, S_total = seller.simulate()

        # print("B:", B)
        if i==0:
            print("B_total_avg:", B_total)
        # print("B_arrival:", B_arrival)
            print("B_duration_avg:", np.mean(B_depart-B_arrival))
        # # print("sorted bids:", sorted(bids, reverse=True))
        # print("=======================")

        # print("S:", S)
        # print("S_total:", S_total)
        # print("S_arrival:", S_arrival)
        # print("S_depart:", S_depart)
        # # print("sorted asks:", sorted(asks, reverse = True))
        # print("=======================")

        # print("BATCH")
        mechanism_batch = mech.Mech_Batch_Welfare(B=B,S=S,B_arrival=B_arrival,S_arrival=S_arrival,B_depart=B_depart,S_depart=S_depart,d=d)
        B_M_batch, S_M_batch = mechanism_batch.run_mechanism()
        M_num_batch += len(B_M_batch)
        W_batch += sum(B_M_batch) - sum(S_M_batch) + sum(S)

        # print("M_num:", len(B_M_batch))
        # print("Welfare:", sum(B_M_batch) + sum(S_M_batch))

        # print("ONLINE")
        mechanism_online = mech.Mech_Online_Welfare(B=B,S=S,B_arrival=B_arrival,S_arrival=S_arrival,B_depart=B_depart,S_depart=S_depart,
            d=d, p_min=price_low, p_max=price_high, alpha=1/2, r=r)
        B_M_online, S_M_online = mechanism_online.run_mechanism()
        M_num_online += len(B_M_online)
        W_online += sum(B_M_online) - sum(S_M_online) + sum(S)
        # print("M_num:", len(B_M_online))
        # print("Welfare:", sum(B_M_online)+sum(S_M_online))
        # print("=======================")

        # print(mechanism.fp_func(1.00738))
        # print(i,"th simulation done")

        mechanism_greedy = mech.Mech_Greedy_Liquidity(B=B,S=S,B_arrival=B_arrival,S_arrival=S_arrival,B_depart=B_depart,S_depart=S_depart)
        B_M_greedy, S_M_greedy = mechanism_greedy.run_mechanism()
        M_num_greedy += len(B_M_greedy)
        W_greedy += sum(B_M_greedy) - sum(S_M_greedy) + sum(S)

    M_num_batch = M_num_batch/N
    W_batch = W_batch/N
    M_num_online = M_num_online/N
    W_online = W_online/N
    M_num_greedy = M_num_greedy/N
    W_greedy = W_greedy/N

    # print("========END=========")
    # print("M_num_batch",M_num_batch)
    # print("M_num_online",M_num_online)
    # print("W_batch",W_batch)
    # print("W_online",W_online)

    return M_num_batch, W_batch, M_num_online, W_online, M_num_greedy, W_greedy

# T = 10
# lam = 5
# period = T

## Simulate bids
price_low = 80
price_high = 120

r = 1.265557

a_rate = 5
d_rate = 2
d = 1
T = 250

kk=10
results = np.empty([kk,6])

for k in range(kk):
    print("k:",k)
    # a_rate = 5*(k+1)
    d = k/2+0.5
    # d_rate = 0.2*k
    results[k,0], results[k,1], results[k,2], results[k,3], results[k,4], results[k,5] = MC_main(a_rate,d_rate,d,T)
print(results)

print(results[:,1]/results[:,3])

plt.plot(range(kk),results[:,1])
plt.plot(range(kk),results[:,3])
plt.plot(range(kk),results[:,5])
plt.legend(['Batch', 'Online', 'Greedy'], loc='upper left')
plt.show()