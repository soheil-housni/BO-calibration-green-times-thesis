import numpy as np
import pandas as pd
from modules.sumo import check_feasibility,update_green_times,SumoSimulation
import sumolib
from lxml import etree
import torch
import sympy as sp
from torch.quasirandom import SobolEngine
import argparse
from modules.sumo import create_gt_dataset
from modules.sumo import TLIncomingCorridors
from modules.sumo import CompleteCorridorsConnexions

def generate_gt_dataset():
    net=sumolib.net.readNet("ingolstadt7/ingolstadt7.net.xml")
    tl_ids=[tl.getID() for tl in net.getTrafficLights()]
    num_replications=10
    cfg_file="ingolstadt7/ingolstadt7.gt_scenario.sumocfg"
    output_dir="gt_data/"


    net_xml="./ingolstadt7/ingolstadt7.net.xml"
    routes_xml="./ingolstadt7/ingolstadt7_routes_with_edges.rou.xml"
    corridors_generator=TLIncomingCorridors(net_xml,routes_xml)
    corridors_df=corridors_generator.get_corridors_df()
    complete_corridors_connexions_generator=CompleteCorridorsConnexions(routes_xml,net_xml,corridors_df)

    complete_corridors_connexions_df=complete_corridors_connexions_generator.create_connexion_corridors_df()
    

    gt_sumo_simulator=SumoSimulation(cfg_file=cfg_file,output_dir=output_dir,tl_ids=tl_ids)
    epoch_index="gt"
    create_gt_dataset(complete_corridors_connexions_df=complete_corridors_connexions_df,
                      output_dir=output_dir,
                      epoch_index=epoch_index,
                      sumo_simulator=gt_sumo_simulator,
                      num_replications=num_replications)


if __name__=="__main__":
    
    generate_gt_dataset()