#!/bin/bash
#$ -S /bin/bash 
#$ -cwd
#$ -jc pcc-skl.72h

date +%F-%T
source ~/.bash_profile
echo `which python`
cd $HOME/shimizulab/batch_script5
python smac_ocm.py > results2_`date +%F-%T`.log
date +%F-%T
