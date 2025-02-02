import numpy as np
import random
import os


def calculate_latency_and_utility(preds_gpu_max_requests, trues, l_time=0.01):
    eps = 1e-6
    # Calculate R where R = ceil(trues / preds_gpu_max_requests)
    R = np.ceil(trues / (preds_gpu_max_requests + eps))

    # Calculate latency for trues <= preds_gpu_max_requests
    latency_case1 = np.full(trues.shape, l_time)  # Latency is l_time when trues <= preds_gpu_max_requests

    # Calculate latency for trues > preds_gpu_max_requests
    M = preds_gpu_max_requests
    latency_sum = np.zeros_like(trues)
    latency_sum = latency_sum.astype(np.float64)
    for r in range(1, int(np.max(R))):  # Loop over possible R values
        # Only update where R > r
        mask = R > r
        latency_sum[mask] = latency_sum[mask] + l_time * r * M[mask]

    # Final latency calculation for the case where trues > preds_gpu_max_requests
    latency_case2 = latency_sum + l_time * R * (trues - (R - 1) * M)
    latency_case2 = latency_case2 / (trues + eps)

    # Combine latencies based on the condition
    latency_all_points = np.where(trues <= preds_gpu_max_requests, latency_case1, latency_case2)

    # Calculate utility for both cases
    utility_case1 = trues / (preds_gpu_max_requests + eps)
    utility_case2 = trues / ((M * R) + eps)

    # Combine utilities based on the condition
    utility_all_points = np.where(trues <= preds_gpu_max_requests, utility_case1, utility_case2)

    # check if the results contains any NaN values
    if np.isnan(latency_all_points).any() or np.isnan(utility_all_points).any():
        raise ValueError("The results contain NaN values")

    return latency_all_points, utility_all_points


def convert_resource(folder_path):
    # 读取预测文件
    # Load the saved 'pred.npy' file
    preds = np.load(folder_path + "pred.npy")
    # the shape of preds is [test_set_length, N, args.pred_len]
    # Load the saved 'true.npy' file
    trues = np.load(folder_path + "true.npy")
    # the shape of trues is [test_set_length, N, args.pred_len]
    preds = np.ceil(preds)
    trues = np.ceil(trues)
    print(f"preds mean: {np.mean(preds)}, var: {np.var(preds)}, min: {np.min(preds)}, max: {np.max(preds)}")
    print(f"trues mean: {np.mean(trues)}, var: {np.var(trues)}, min: {np.min(trues)}, max: {np.max(trues)}")
    # Step 1: Build a random seed list containing 5 random seeds
    random_seed_list = [100, 200, 300, 400, 500]
    # Step 2: Define N
    N = trues.shape[1]  # Replace with your desired value of N
    # Step 3: Define the list of possible number of requests that can be handled by a single GPU
    if "ALI" in folder_path:
        possible_values = [2**i for i in range(40, 50)]
        print("using possible_values for ALI", possible_values)
    else:
        possible_values = [2**i for i in range(10, 30)]

    utility_collection = []
    latency_collection = []

    # Step 4: Repeat the procedure 5 times
    for seed_value in random_seed_list:
        print(f"Running simulation for seed value: {seed_value}")
        # print(f"Running simulation for seed value: {seed_value}")
        # Initialize the random seed for both Python's random module and NumPy
        random.seed(seed_value)
        np.random.seed(seed_value)

        # Step 5: Initialize a numpy array with the shape (N,)
        # This array will store the number of requests that can be handled by each gpu for each service
        gpu_maximum_requests = np.random.choice(possible_values, size=N)
        # the shape of gpu_maximum_requests is (N,)
        # Convert the requests in preds and trues to the number of GPUs needed
        preds_gpu = np.maximum(1.0, np.ceil(preds / gpu_maximum_requests[:, np.newaxis]))
        # the shape of preds_gpu is [test_set_length, N, args.pred_len]
        trues_gpu = np.ceil(trues / gpu_maximum_requests[:, np.newaxis])
        # the shape of trues_gpu is [test_set_length, N, args.pred_len]

        # Convert preds_gpu into preds_gpu_max_requests using the suggested method
        preds_gpu_max_requests = gpu_maximum_requests.reshape(1, N, 1) * preds_gpu
        # the shape of preds_gpu_max_requests is [test_set_length, N, args.pred_len]

        # Calculate the latency and utility for each service at each time step
        latency_all_points, utility_all_points = calculate_latency_and_utility(preds_gpu_max_requests, trues)
        # the shape of latency_all_points and utility_all_points is [test_set_length, N, args.pred_len]
        # Append the results to the respective collections
        latency_collection.append(latency_all_points)
        utility_collection.append(utility_all_points)
        # the type of latency_collection and utility_collection is list
        # each element in the list is a numpy array of shape [test_set_length, N, args.pred_len]

    # Convert the list of numpy arrays into a single numpy array
    latency_collection_np = np.array(latency_collection)
    utility_collection_np = np.array(utility_collection)
    # the shape of latency_collection_np and utility_collection_np is [5, test_set_length, N, args.pred_len]
    # Average along the first dimension (which corresponds to the different seeds)
    average_seed_latency = np.mean(latency_collection_np, axis=0)
    average_seed_utility = np.mean(utility_collection_np, axis=0)
    # the shape of average_seed_latency and average_seed_utility is [test_set_length, N, args.pred_len]

    # we want to average the results over all dimensions
    final_latency = np.mean(average_seed_latency)
    final_utility = np.mean(average_seed_utility)

    return final_latency, final_utility


if __name__ == "__main__":
    # Define the folder path where the predictions are saved
    folder_path = "../test_results/0_forecast_train_ccformer_APP_RPC_ftS_sl168_ll1_pl1_dm32_nh8_el2_dl1_df64_fc1_eblearned_dtTrue_test_0_freq_1min/"
    # Call the function to convert the resource
    final_latency, final_utility = convert_resource(folder_path)
    print(f"Final latency: {final_latency}, Final utility: {final_utility}")