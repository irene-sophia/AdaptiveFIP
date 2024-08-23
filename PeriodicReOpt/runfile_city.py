import pandas as pd
import sys

from optimize_city import run_instance

if __name__ == '__main__':
    results_df = pd.DataFrame(columns=['n_realizations', 'num_units', 'num_sensors', 'instance', 'pct_intercepted'])

    # num_units = int(sys.argv[-3])
    # num_sensors = int(sys.argv[-2])
    # n_realizations = int(sys.argv[-1])
    num_units = 3
    num_sensors = 3
    n_realizations = 100

    graph_type = 'city'
    city = 'Manhattan'
    t_max = 1800

    for instance in range(1):

        pct_intercepted = run_instance(graph_type, city, n_realizations, num_units, num_sensors, instance)

        list_row = [n_realizations, num_units, num_sensors, instance, pct_intercepted]
        results_df.loc[len(results_df)] = list_row
        results_df.to_csv(f'results/city/results_{city}_R{n_realizations}_U{num_units}_S{num_sensors}_upto{instance}.csv', index=False)

    print(results_df)

    results_df.to_csv(f'results/city/results_{city}_R{n_realizations}_U{num_units}_S{num_sensors}.csv', index=False)