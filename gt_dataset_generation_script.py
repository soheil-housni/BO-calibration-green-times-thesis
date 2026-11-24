import numpy as np
import pandas as pd
import sumolib
from lxml import etree
import torch
import sympy as sp
from torch.quasirandom import SobolEngine
import argparse
from modules.sumo import create_gt_dataset

def generate_gt_dataset():
    net=sumolib.net.readNet("ingolstadt7/ingolstadt7.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    num_replications=10
    cfg_file="ingolstadt7/ingolstadt7.gt_scenario.sumocfg"
    output_dir="gt_data/"
    corridors_df=torch.load("./corridors_savings/corridors_df.pt",weights_only=False)
    epoch_index="gt"
    create_gt_dataset(corridors_df=corridors_df,
                      output_dir=output_dir,
                      epoch_index=epoch_index,
                      num_replications=num_replications,
                      cfg_file=cfg_file,
                      tl_ids=tl_ids)



if __name__=="__main__":
    
    generate_gt_dataset()