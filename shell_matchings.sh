#!/bin/sh
#
#$ -M tchin@cs.brown.edu
#$ -m abes
#
#  Execute from the current working directory
#$ -cwd
#
#  This is a long-running job
#$ -l inf
#
#  Can use up to 6GB of memory
#$ -l vf=16G
#
python box_matchings.py
