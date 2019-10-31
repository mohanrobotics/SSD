#!/bin/bash
#SBATCH --partition gpu_titan          # partition (queue)
#SBATCH --nodes 1                # number of nodes
#SBATCH --mem 15g               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 3-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output output.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error error.%N.%j.err  # filename for STDERR
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1

source /gluster/home/mmuthuraja/anaconda3/bin/activate ssd_pytorch

