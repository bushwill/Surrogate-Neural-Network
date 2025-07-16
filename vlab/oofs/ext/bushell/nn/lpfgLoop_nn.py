#!/usr/bin/python

import os
import shutil
import glob
import time
import subprocess
import signal
import csv
import sys
import numpy
from numpy.random import normal as nran
from numpy.random import uniform as uran

# Assumes directory lpfg_run_(run #) is set up before calling
def generatePlant(run = 1, runs = 10):
    for i in range(runs):
        sys.stdout.write("progress: %g%%   \r" % (100. * float(i) / float(runs)) )
        sys.stdout.flush()
        # setup call to lpfg
        # lpfg_command = "lpfg -w 306 256 lsystem.l view.v materials.mat -a anim.a contours.cset functions.fset functions.tset loop_parameters.vset > log.txt"
        lpfg_command = f"lpfg -w 306 256 lsystem.l view.v materials.mat contours.cset functions.fset functions.tset lpfg_run_{run+1}/run_{i+1}/run_parameters.vset > lpfg_run_{run+1}/run_{i+1}/lpfg_log.txt"

        if not os.path.exists("project"):
            os.system("g++ -o project -Wall -Wextra project.cpp -lm")

        # run lpfg  
        process = subprocess.Popen(['bash', '-c', lpfg_command])
        process.wait()

        os.system(f"./project 2454 2056 leafposition.dat > lpfg_run_{run+1}/run_{i+1}/output.txt")
        shutil.move("leafposition.dat", f"./lpfg_run_{run+1}/run_{i+1}")
        
