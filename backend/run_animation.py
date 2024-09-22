import numpy as np
import csv
import arrow
from animation.AnimData import load_data
from animation.AnimEvaluator import *

from animation.algorithm.RouteFlowAnim import RouteFlowAnim

if __name__ == '__main__':
    anim_list = [RouteFlowAnim()]
        
    # ---------- Prepare for Dataset ----------
    
    dataset_list = [
        'Taxi',
        'BirdMap',
        'Railway',
        'MEIBook',
        'OpenSkyAirline',
        'USMigration',
        'DanishAIS',
    ]

    # ---------- Prepare for Metrics ----------
    r = 6.0 / 1024
    metrics = [
        # animation metric
        WithinGroupOcclusionMetric(threshold_occlusion=r*0.9),
        OverallOcclusionMetric(threshold_occlusion=r*0.9),
        DispersionMetric(threshold_occlusion=r),
        DeformationMetric(threshold_occlusion=r),
    ]


    # ---------- Pipeline Running -----------
    metrics_results = np.zeros((len(anim_list), len(metrics)))

    now_time = arrow.now().format('YYYY-MM-DD_HH-mm-ss')
    csv_path = f'./log/metrics/Metrics_{now_time}.csv'

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        metrics_name = [metric.name for metric in metrics]
        attr_row = ['Algorithm', 'Dataset'] + metrics_name
        writer.writerow(attr_row)
    total = len(anim_list) * len(dataset_list)
    progress = 0
    for i, anim in enumerate(anim_list):
        for j,dataset_name in enumerate(dataset_list):
            progress += 1
            print(f'Progress: {progress}/{total}')
            path_input = load_data(dataset_name)
            print(f'dataset_name={dataset_name}')

            anim.set_input(dataset_name, path_input)

            output = anim.get_output()


            groups = anim.anime_groups

            print(f'Animating {dataset_name}')
            anim.show(dataset_name, points_to_trace=[])

            csv_row = [anim.algo_name, dataset_name]
            for j, metric in enumerate(metrics):
                print(f'Computing {metric.name}')
                metric.set_debug_name(f'{dataset_name}_{anim.algo_name}')
                try: 
                    val = metric(output, groups, anim.anime_start, anim.anime_end)
                except Exception:
                    val = 9999.0
               
                csv_row.append(val)
                metrics_results[i, j] += val

            # round
            for j in range(len(csv_row)):
                try:
                    csv_row[j] = round(csv_row[j], 6)
                except Exception:
                    pass
            
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(csv_row)
    
    metrics_results /= len(dataset_list)

    for i, anim in enumerate(anim_list):
        csv_row = [anim.algo_name, 'average']
        for j, metric in enumerate(metrics):
            csv_row.append(metrics_results[i, j])
        csv_row.append(metrics_results[i, -1])
        
        # round
        for j in range(len(csv_row)):
            try:
                csv_row[j] = round(csv_row[j], 6)
            except Exception:
                pass

        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_row)
