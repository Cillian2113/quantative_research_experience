import pandas as pd
import numpy as np

loan_data = pd.read_csv('Task 3 and 4_Loan_Data.csv')
scores = np.sort(loan_data["fico_score"])

def quantize_fico(K):
    
    values, counts = np.unique(scores, return_counts=True)
    n = len(values)
    
    # We compute SSE for every interval of scores
    sse = np.zeros((n, n))
    for start in range(n):
        score_sum = 0
        total_count = 0
        for end in range(start, n):
            score_sum += values[end] * counts[end]
            total_count += counts[end]
            mean = score_sum / total_count
            sse[start][end] = np.sum(counts[start:end+1] * (values[start:end+1] - mean)**2)
    
    # DP table: dp[i][k] will be the minimum SSE for first i scores into k buckets
    dp = np.full((n+1, K+1), np.inf)
    #SSE for 0 scores and zero buckets is zero
    dp[0][0] = 0
    #bucket_index will give the index in scores where the last bucket starts for each i k combination
    bucket_index = np.zeros((n+1, K+1), dtype=int)
    
    #We find the optimal bucket arrangement for with the first scores and then use dynamic programming whilst adding more scores to find the optimal arrangement
    for i in range(1, n+1):
        for k in range(1, min(i, K)+1):
            for j in range(k-1, i):
                #cost is minimum SSE achievable with j scores and k-1 buckets + the SSE from if we put the bucket at index j
                cost = dp[j][k-1] + sse[j][i-1]
                #If this is the best SSE found so far we store the cost to beat and the index at where the last bucket should now be placed
                if cost < dp[i][k]:
                    dp[i][k] = cost
                    bucket_index[i][k] = j

    # Backtrack through the table to find bucket boundaries
    boundaries = []
    i, k = n, K
    while k > 0:
        j = bucket_index[i][k]
        boundaries.append((values[j], values[i-1]))
        i = j
        k = k-1
    boundaries.reverse()
    
    return boundaries

def total_mse(bucket_boundaries):
    total = 0
    for score in scores:
        for (low, high) in bucket_boundaries:
            if low <= score <= high:
                bucket_mid = (low + high) / 2
                total += (score - bucket_mid) ** 2
                break
    mse = total / len(scores)
    return mse


if __name__ == "__main__":
    K = 5  
    boundaries = quantize_fico(K)
    boundaries = [(int(low), int(high)) for low, high in boundaries]

    n = len(scores)
    quantile_indices = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n]
    quantile_boundaries = []
    bucket_mean = []
    for i in range(5):
        bucket_scores = scores[quantile_indices[i]:quantile_indices[i+1]]
        min_score = bucket_scores[0]
        max_score = bucket_scores[-1]
        quantile_boundaries.append((min_score, max_score))
    quantile_boundaries = [(int(low), int(high)) for low, high in quantile_boundaries]
    print(total_mse(quantile_boundaries))
    print(total_mse(boundaries))









