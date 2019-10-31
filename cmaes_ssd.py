import argparse
import cma
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import subprocess
import yaml
from multiprocessing import Pool
import time

def is_float(test_string):
    # Check for float string
    try:
        float(test_string)
        res = True
    except:
        res = False
    return res

def get_mAP(file_path):

    with open(file_path, 'r') as f:
        for line in f:
            if 'mAP' in line:
                for s in line.split(":"):
                    if is_float(s):
                        return float(s)

def ssd(ips):
    x = ips[0]
    itr = ips[1]
    gene_no = ips[2]

    key = 'Itr_' + str(itr) + '_gene' + str(gene_no)

    print('current i/p',x)
    anchor_scales = [(300 * anc).item() for anc in x[:7]]
    min_sizes = anchor_scales[:6]
    max_sizes = anchor_scales[1:7]

    print(type(min_sizes[0]))
    # anchor_ratios = [[x[7]], [x[7], x[8]], [x[7], x[8]], [x[7], x[8]], [x[7]], [x[7]]]


    # Opening the cfg template and writing the config template
    stream = open('templates/cfg_template.yml', 'r')
    cfg_data = yaml.safe_load(stream)
    cfg_data['MODEL']['PRIORS']['MIN_SIZES'] = min_sizes
    cfg_data['MODEL']['PRIORS']['MAX_SIZES'] = max_sizes
    # cfg_data['MODEL']['PRIORS']['ASPECT_RATIOS'] = anchor_ratios
    cfg_data['OUTPUT_DIR'] = 'outputs/{}_vgg_ssd300_voc07'.format(key)


    with open(
            'configs/' + '{}_vgg_ssd300_voc07'.format(key) + '.yaml',
            'w'
    ) as outfile:
        yaml.dump(cfg_data, outfile, default_flow_style=False)



    shutil.copy(
        'templates/{}'.format('bash_file_template.sh'),
        'bash_files/{}'.format(key) +  '.sh'
    )


    job_file = 'bash_files/{}'.format(key) +  '.sh'

    with open(job_file, "a") as myfile:
        myfile.write(
            "\nsrun "
            "--ntasks 1 "
            "--job-name job_trial "
            "--mem-per-cpu 15g "
            "--cpus-per-task 1 "
            "--partition gpu_titan "
            "--gres gpu:1 "
            "python train.py "
            "--config-file configs/{}_vgg_ssd300_voc07.yaml".format(key)
        )

    # Running the job file
    os.system('sbatch -W %s' % job_file)

    result_file = 'outputs/{}_vgg_ssd300_voc07/inference/voc_2007_test/result_final.txt'.format(key)
    mAP = get_mAP(result_file)
    print("map is", mAP)
    return 1 - mAP

def CMA_ES():

    previous_cmaes_path = None
    run_from_previous = False

    max_evaluations = 18
    initial_guess = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    sigma = 0.1
    bounds = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06]
    ]

    cma_obj_path = 'objects/'

    if run_from_previous:
        es = pickle.load(open('_saved-cma-object.pkl', 'rb'))
    else:
        es = cma.CMAEvolutionStrategy(initial_guess, sigma, {'bounds': bounds})

    itr = 0
    max_itr = int(max_evaluations / es.popsize)
    print('Population size is:', es.popsize)

    solution_dict = {i: [] for i in range(max_itr)}
    results_dict = {i: [] for i in range(max_itr)}

    best_gene = initial_guess
    best_value = float("inf")

    while itr < max_itr:

        print('Iteration {} is Running:'.format(itr+1) )
        # getting the genes for evaluation
        solutions = es.ask()

        # evaluating the genes
        with Pool(es.popsize + 1 ) as p:
            fn_results = p.map(ssd, [[s,itr,i] for i,s in enumerate(solutions)])
        # fn_results = [frosenbrock(s) for s in solutions]

        print('Solutions: ', solutions)
        print('Results: ',fn_results)
        # optimising based on the results
        es.tell(solutions, fn_results)

        # logging the data for results
        es.logger.disp_header()
        es.disp()
        es.logger.add()

        # Appendind the datas
        solution_dict[itr].append(solutions)
        results_dict[itr].append(fn_results)

        # checking for best gene
        if min(fn_results) < best_value:
            best_value = np.min(fn_results)
            best_gene = solutions[np.argmin(fn_results)]
        itr = itr + 1

        pickle.dump(es, open('{}saved-cma-object.pkl'.format(cma_obj_path), 'wb'))
        pickle.dump(solution_dict, open('{}solution_dict.pkl'.format(cma_obj_path), 'wb'))
        pickle.dump(results_dict, open('{}results_dict.pkl'.format(cma_obj_path), 'wb'))

CMA_ES()
