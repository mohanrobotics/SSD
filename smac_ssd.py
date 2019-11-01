import logging
import yaml
import numpy as np
import os
import argparse

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac.initial_design.default_configuration_design import DefaultConfiguration

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_bo_facade import SMAC4BO

import shutil
import subprocess


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


def ssd(x):

    print('current i/p',x)

    # counter for generating key
    with open(counter_filename, "a") as myfile:
        myfile.write("1\n")

    # Number of line in a the file is taken as a key
    key = len(open(counter_filename).readlines())
    key = 'ssd_' + str(key) + '_' + arg.root_key

    anchor_scales = [
        300 * (
            x['anchor_scale_{}'.format(i)]
        ) for i in range(7)
    ]

    min_sizes = anchor_scales[:6]
    max_sizes = anchor_scales[1:7]

    # Opening the cfg template and writing the config template
    stream = open('templates/cfg_template.yml', 'r')
    cfg_data = yaml.safe_load(stream)
    cfg_data['MODEL']['PRIORS']['MIN_SIZES'] = min_sizes
    cfg_data['MODEL']['PRIORS']['MAX_SIZES'] = max_sizes
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
            "--nodes 1 "
            "--job-name job_trial "
            "--mem-per-cpu 40g "
            "--cpus-per-task 4 "
            "--partition gpu4 "
            "--gpu 4 "
            "python -m torch.distributed.launch "
            "--nproc_per_node=4 "
            "train.py "
            "--config-file configs/{}_vgg_ssd300_voc07.yaml "
            "SOLVER.WARMUP_FACTOR 0.03333 "
            "SOLVER.WARMUP_ITERS 1000 "
            ">> ../results/res_{}.txt\n ".format(key,key)
        )

    # Running the job file
    os.system('sbatch -W %s' % job_file)

    result_file = 'outputs/{}_vgg_ssd300_voc07/inference/voc_2007_test/result_final.txt'.format(key)
    mAP = get_mAP(result_file)
    print("map is", mAP)
    return 1 - mAP


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='SMAC/BO for faster ssd'
    )
    # parser.add_argument(
    #     'cluster',
    #     type=str,
    #     help='cluster to use uni/dfki/dfkinode2'
    # )
    # parser.add_argument(
    #     'partition',
    #     type=str,
    #     help='cluster partition'
    # )

    parser.add_argument(
        'opt_method',
        type=str,
        help='optimization method enter SMAC/BO'
    )

    parser.add_argument(
        'root_key',
        type=str,
        help='key for saving job names and cfgs'
    )

    arg = parser.parse_args()

    counter_filename = 'counter_{}.txt'.format(arg.root_key)

    # Create a file for counter key generation
    open(counter_filename, 'a').close()


    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()

    anchor_scale_0 = UniformFloatHyperparameter("anchor_scale_0", 0.0, 1.06, default_value=0.1)
    anchor_scale_1 = UniformFloatHyperparameter("anchor_scale_1", 0.0, 1.06, default_value=0.2)
    anchor_scale_2 = UniformFloatHyperparameter("anchor_scale_2", 0.0, 1.06, default_value=0.37)
    anchor_scale_3 = UniformFloatHyperparameter("anchor_scale_3", 0.0, 1.06, default_value=0.54)
    anchor_scale_4 = UniformFloatHyperparameter("anchor_scale_4", 0.0, 1.06, default_value=0.71)
    anchor_scale_5 = UniformFloatHyperparameter("anchor_scale_5", 0.0, 1.06, default_value=0.88)
    anchor_scale_6 = UniformFloatHyperparameter("anchor_scale_6", 0.0, 1.06, default_value=1.05)

    cs.add_hyperparameters([
        anchor_scale_0,
        anchor_scale_1,
        anchor_scale_2,
        anchor_scale_3,
        anchor_scale_4,
        anchor_scale_5,
        anchor_scale_6
    ])


    # Scenario object
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": 75,   # max. number of function evaluations; for this example set to a low number
                         "cs": cs,               # configuration space
                         "deterministic": "true"
                         })


    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    if arg.opt_method == 'SMAC':
        smac = SMAC4HPO(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=ssd,
            initial_design=DefaultConfiguration
        )
    elif arg.opt_method == 'BO':
        smac = SMAC4BO(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=ssd,
            initial_design=DefaultConfiguration
        )
    smac.optimize()
    # deleting the entries in the counter
    with open(counter_filename, 'w'): pass
