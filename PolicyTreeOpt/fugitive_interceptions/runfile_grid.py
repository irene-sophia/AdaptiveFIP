import pandas as pd
import sys

from sequential_runfile_sorted_filtered import run_instance

if __name__ == '__main__':
    results_df = pd.DataFrame()

    U = int(sys.argv[-3])
    num_sensors = int(sys.argv[-2])
    R = int(sys.argv[-1])

    N = 10
    T = int(5 + (0.5 * N))
    num_instances = 10
    num_seeds = 10

    for instance in range(num_instances):
        results_instance = run_instance(N, U, R, num_sensors, num_seeds)

        results_df = pd.concat([results_df, results_instance])

    results_df.to_csv(f'results/manhattan/results_N{N}_R{R}_U{U}_S{num_sensors}.csv', index=False)