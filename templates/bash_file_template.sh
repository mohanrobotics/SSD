#!/bin/bash
#SBATCH --partition gpu4          # partition (queue)
#SBATCH --nodes 1                # number of nodes
#SBATCH --mem 160g               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 3-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output ssd_4.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error ssd_4.%N.%j.err  # filename for STDERR
#SBATCH --gpus 4
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 40g

module load cuda
source /home/mmuthu2s/anaconda3/bin/activate pytorch