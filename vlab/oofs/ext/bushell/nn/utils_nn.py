from numpy.random import normal as nran
from numpy.random import uniform as uran
import os
import subprocess
import shutil
from plant_comparison_nn import calculate_cost, read_real_plants

def get_normalization_stats():
    """
    Returns normalization statistics for the surrogate model.
    Adjust these values according to your dataset.
    For inputs (dimension 13) and a scalar output.
    """
    import torch
    input_mean = torch.zeros(13)
    input_std = torch.ones(13)
    output_mean = torch.tensor(0.0)
    output_std = torch.tensor(1.0)
    return input_mean, input_std, output_mean, output_std

def generate_plant(param_file, output_dir):
    """
    Generate a plant using lpfg and save results in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lpfg_command = f"lpfg -w 306 256 lsystem.l view.v materials.mat contours.cset functions.fset functions.tset {param_file} > {output_dir}/lpfg_log.txt"
    # Compile project if needed
    if not os.path.exists("project"):
        os.system("g++ -o project -Wall -Wextra project.cpp -lm")
    # Run lpfg
    process = subprocess.Popen(['bash', '-c', lpfg_command])
    process.wait()
    os.system(f"./project 2454 2056 leafposition.dat > {output_dir}/output.txt")
    shutil.move("leafposition.dat", f"./{output_dir}")

def read_syn_plant(file_name):
    """
    Read endpoints and branchpoints from a plant output file.
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
    day_temp = 0
    syn_bp = []
    syn_ep = []
    syn_bp_day = []
    syn_ep_day = []
    day = []
    for line in lines:
        temp = line.split(" ")
        if temp[0] == "Day:":
            day_temp = int(temp[1])
            if day_temp > 2:
                syn_bp.append(syn_bp_day)
                syn_ep.append(syn_ep_day)
                syn_bp_day = []
                syn_ep_day = []
        if (temp[0] != "Day:") & (day_temp > 1):
            if temp[0] == "I":
                syn_bp_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)
            else:
                syn_ep_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)
    if day_temp == 27:
        syn_bp.append(syn_bp_day)
        syn_ep.append(syn_ep_day)
    return syn_bp, syn_ep

def generate_and_evaluate_in_dir(param_file, real_bp, real_ep, output_dir, cost_fn):
    """
    Generate a plant in output_dir and evaluate its cost using cost_fn.
    """
    generate_plant(param_file, output_dir)
    syn_bp, syn_ep = read_syn_plant(f"{output_dir}/output.txt")
    cost = 0
    for i in range(min(len(syn_bp), len(real_bp))):
        cost += cost_fn(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
    return cost

def build_parameter_file(filename, params):
    with open(filename, "w") as f:
        f.write(f"#define MAX_PHYTOMERS {params[0]}\n")
        f.write(f"#define PLASTOCHRON {params[1]}\n")
        f.write(f"#define PlantRollAng {params[2]}\n")
        f.write(f"#define PlantDownAng {params[3]}\n")
        f.write(f"#define BrAngle {params[4]}\n")
        f.write(f"#define LeafLen {params[5]}\n")
        f.write(f"#define ExpLeafWid {params[6]}\n")
        f.write(f"#define LeafWid {params[7]}\n")
        f.write(f"#define LEAF_BEND_SCALE {params[8]}\n")
        f.write(f"#define LEAF_TWIST_SCALE {params[9]}\n")
        f.write(f"#define IntLen {params[10]}\n")
        f.write(f"#define IntWid {params[11]}\n")
        f.write(f"#define ExpIntRad {params[12]}\n")
        
def build_random_parameter_file(dir_name):
    f = open(dir_name, "w")
    max_phy = nran(10.,1.)
    plast = nran(3.,0.1)
    chirality = 1.
    if uran(0.,1.) < 0.5 :
        chirality = -1.
    plant_roll_angle = nran(chirality * 90.,10.0)
    plant_down_angle = nran(0.,4.0)
    branch_angle = nran(135.,5.)
    leaf_len = nran(5.,1.)
    exp_leaf_wid = nran(0.5,0.01)
    leaf_wid = nran(1.,0.1)
    leaf_bend_scale = nran(90.,3.)
    leaf_twist_scale = nran(180.,3.)
    node_len = nran(0.7,0.05)
    int_wid = nran(0.9,0.01)
    exp_int_rad = nran(0.5,0.01)
    f.write('#define MAX_PHYTOMERS ' + str(max_phy) + '\n')
    f.write('#define PLASTOCHRON ' + str(plast) + '\n')
    f.write('#define PlantRollAng ' + str(plant_roll_angle) + '\n')
    f.write('#define PlantDownAng ' + str(plant_down_angle) + '\n')
    f.write('#define BrAngle ' + str(branch_angle) + '\n')
    f.write('#define LeafLen ' + str(leaf_len) + '\n')
    f.write('#define ExpLeafWid ' + str(exp_leaf_wid) + '\n')
    f.write('#define LeafWid ' + str(leaf_wid) + '\n')
    f.write('#define LEAF_BEND_SCALE ' + str(leaf_bend_scale) + '\n')
    f.write('#define LEAF_TWIST_SCALE ' + str(leaf_twist_scale) + '\n')
    f.write('#define IntLen ' + str(node_len) + '\n')
    f.write('#define IntWid ' + str(int_wid) + '\n')
    f.write('#define ExpIntRad ' + str(exp_int_rad) + '\n')
    f.close()
    return [max_phy, plast, plant_roll_angle, plant_down_angle, branch_angle, leaf_len, exp_leaf_wid, leaf_wid, leaf_bend_scale, leaf_twist_scale, node_len, int_wid, exp_int_rad]

def generate_and_evaluate(param_file, real_bp, real_ep):
    # Run lpfg to generate the synthetic plant
    generateSurrogatePlant(param_file)
    # Read the synthetic plant's endpoints and branchpoints for the latest run
    syn_bp, syn_ep = read_syn_plant_surrogate()
    # Use the first (or only) day's data for cost calculation
    cost = 0
    for i in range(min(len(syn_bp), len(real_bp))):
        cost += calculate_cost(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
    return cost

def generateSurrogatePlant(param_file):
        # setup call to lpfg
        # lpfg_command = "lpfg -w 306 256 lsystem.l view.v materials.mat -a anim.a contours.cset functions.fset functions.tset loop_parameters.vset > log.txt"
        lpfg_command = f"lpfg -w 306 256 lsystem.l view.v materials.mat contours.cset functions.fset functions.tset {param_file} > surrogate/lpfg_log.txt"

        if not os.path.exists("project"):
            os.system("g++ -o project -Wall -Wextra project.cpp -lm")
            
        if not os.path.exists("surrogate"):
            os.mkdir("surrogate")

        # run lpfg  
        process = subprocess.Popen(['bash', '-c', lpfg_command])
        process.wait()
        os.system(f"./project 2454 2056 leafposition.dat > surrogate/output.txt")
        shutil.move("leafposition.dat", f"./surrogate")
        
def read_syn_plant_surrogate(file_name="surrogate/output.txt"):
    f = open(file_name, "r")
    lines = f.readlines()
    day_temp = 0
    syn_bp = []
    syn_ep = []
    syn_bp_day = []
    syn_ep_day = []
    day = []

    for line in lines:
        temp = line.split(" ")
        if temp[0] == "Day:":
            day_temp = int(temp[1])
            if day_temp>2:
                syn_bp.append(syn_bp_day)
                syn_ep.append(syn_ep_day)
                syn_bp_day = []
                syn_ep_day = []
        if (temp[0] != "Day:") & (day_temp > 1):
            if temp[0] == "I":
                syn_bp_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)
            else:
                syn_ep_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)

    if day_temp == 27:
        syn_bp.append(syn_bp_day)
        syn_ep.append(syn_ep_day)

    f.close()

    return syn_bp, syn_ep