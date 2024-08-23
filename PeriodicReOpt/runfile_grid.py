
import pandas as pd

from optimize_grid import run_instance

if __name__ == '__main__':
    results_df = pd.DataFrame(columns=['manhattan_diameter', 'n_realizations', 'num_units', 'num_sensors', 'instance', 'pct_intercepted'])

    num_units = 10
    num_sensors = 10
    n_realizations = 10

    graph_type = 'manhattan'
    manhattan_diam = 10
    t_max = int(5 + (0.5 * manhattan_diam))

    for instance in range(10):

        pct_intercepted = run_instance(graph_type, n_realizations, manhattan_diam, num_units, num_sensors, instance)

        list_row = [manhattan_diam, n_realizations, num_units, num_sensors, instance, pct_intercepted]
        results_df.loc[len(results_df)] = list_row

    print(results_df)
    results_df.to_csv(f'results/manhattan/results_N{manhattan_diam}_R{n_realizations}_U{num_units}_S{num_sensors}.csv', index=False)