import json
import numpy as np
from collections import defaultdict

from tqdm import tqdm


num_experts = 128
top_k = 8
batch_size = 16


def generate_random_indices(batch_size, num_experts, top_k):
    """
    Generate random indices for top-k selection from a set of experts.

    Args:
        batch_size (int): Number of samples in the batch.
        num_experts (int): Total number of experts.
        top_k (int): Number of top experts to select.

    Returns:
        np.ndarray: Randomly selected indices of shape (batch_size, top_k).
    """
    return np.random.choice(num_experts, size=(batch_size, top_k), replace=True)


def main():
    indices = generate_random_indices(batch_size, num_experts, top_k)
    print("Randomly selected indices for top-k experts:")
    print(f"nuber of unique experts: {np.unique(indices).size / num_experts * 100:.2f}%")

    ratios = []
    num_selected_experts = []
    arrs = defaultdict(list)
    with open("out.log", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if line.startswith("[DEBUG]"):
                try:
                    arr = json.loads(line[len("[DEBUG]"):].strip())
                    arr = np.array(arr)
                    values, counts = np.unique(arr, return_counts=True)
                    num_selected_experts.append(counts.sum() // top_k)
                    if abs(counts.sum() // top_k - 128) <= 0:
                        arrs[counts.sum() // top_k].append(values)
                        print(f"Number of unique experts in the log: {values.size / num_experts * 100:.2f}%")
                        ratios.append(values.size / num_experts * 100)

                except json.JSONDecodeError:
                    pass

    print(f"Average ratio of unique experts in the log: {np.mean(ratios):.2f}%")
    len_arrs = len(arrs[128])

    ratios = []
    for i in range(len_arrs // 4):
        values_0 = arrs[128][i]
        values_1 = arrs[128][len_arrs // 4 + i]
        values_2 = arrs[128][len_arrs // 4 * 2 + i]
        values_3 = arrs[128][len_arrs // 4 * 3 + i]
        values = np.concatenate([values_0, values_1, values_2, values_3])
        unique_values, counts = np.unique(values, return_counts=True)
        print(f"Number of unique experts in the log for {i} and {len_arrs // 2 + i}: {unique_values.size / num_experts * 100:.2f}%")
        ratios.append(unique_values.size / num_experts * 100)

    print(f"Average ratio of unique experts in the log for pairs: {np.mean(ratios):.2f}%")
    print(f"Average number of selected experts: {np.mean(num_selected_experts)}")


if __name__ == "__main__":
    main()
