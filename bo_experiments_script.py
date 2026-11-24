from modules.sumo import ErrorFunctionRouteVersion,TLIncomingCorridors,update_green_times,SumoSimulation,create_initial_tl_df
from modules.gp import UpdatedFAFunctionLinearRegresion,InitializeGPModel
from modules.bo import InitialSamples,Optimization,BOLoop
import pandas as pd
import numpy as np
import os
import shutil
import random
import argparse
from itertools import product
import torch
from lxml import etree
import ast



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_sumo_files(kernel_name,mean_name,annotation,rs):

    set_seed(rs)

    network_dir = "./ingolstadt7"

    original_additional = os.path.join(
        network_dir, "additional.xml"
    )

    original_config = os.path.join(
        network_dir, "ingolstadt7.sumocfg"
    )

    additional_xml = os.path.join(
        network_dir, f"additional_{kernel_name}_{mean_name}_{annotation}_rs{rs}.xml"
    )

    cfg_file = os.path.join(
        network_dir, f"ingolstadt7_{kernel_name}_{mean_name}_{annotation}_rs{rs}.sumocfg"
    )

    shutil.copy2(original_additional, additional_xml)
    shutil.copy2(original_config, cfg_file)

    tree = etree.parse(cfg_file)

    additional_files = tree.find(".//additional-files")

    if additional_files is None:
        raise ValueError("additional-files introuvable dans le sumocfg")

    additional_files.set("value", f"additional_{kernel_name}_{mean_name}_{annotation}_rs{rs}.xml")

    tree.write(cfg_file, encoding="UTF-8", xml_declaration=True)

    return cfg_file, additional_xml




def run_bo(kernel_name,mean_name,id_init_sample,rs):
    set_seed(rs)

    x_dim=7
    output_dir = 'BO_outputs/'

    df_gt=pd.read_csv("gt_data/df_gt_InTAS7DIM.csv")
    df_gt["route"]=df_gt["route"].apply(lambda x : ast.literal_eval(x))
    df_gt=df_gt.sort_values(by="route_id")

    scaling_loss_factor=1

    gt_additional_xml="ingolstadt7/additional.gt_scenario.xml"
    initial_tl_df=create_initial_tl_df(gt_additional_xml)

    tl_ids = np.sort(initial_tl_df["tl"].unique())


    low_bound = 5
    up_bound = 74
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    x_bounds = torch.tensor([[low_bound]*x_dim,
                        [up_bound]*(x_dim-1)+[50]], 
                        device=device, dtype=dtype)

    annotation=f"nu_2.5_route_error_route_file_original_is{id_init_sample}"
    cfg_file, additional_xml = prepare_sumo_files(kernel_name=kernel_name,mean_name=mean_name,annotation=annotation,rs=rs)


    sumo_simulator=SumoSimulation(
        cfg_file=cfg_file,
        output_dir=output_dir,
        tl_ids=tl_ids
    )


    net_xml="./ingolstadt7/ingolstadt7.net.xml"
    route_xml="./ingolstadt7/ingolstadt7_routes_with_edges.rou.xml"
    corridors_generator=TLIncomingCorridors(net_xml,route_xml)
    corridors_df=corridors_generator.get_corridors_df()

    dtype=torch.double



    error_function=ErrorFunctionRouteVersion(initial_tl_df=initial_tl_df,
                                                 df_gt=df_gt,
                                                 sumo_simulator=sumo_simulator,
                                                 corridors_df=corridors_df,
                                                 additional_xml=additional_xml,
                                                 output_dir=output_dir,
                                                 tl_ids=tl_ids)
    

    fA_function=UpdatedFAFunctionLinearRegresion(df_gt=df_gt,initial_tl_df=initial_tl_df,net_file=net_xml)

    gp_initializer=InitializeGPModel(
        fa_function=fA_function,
        bounds=x_bounds,
        dtype=dtype,
        rbf=False,
        nu=2.5,
        mean_mod=True
    )

    batch_size=1
    raw_samples=512
    num_restarts_opti=5

    optimizer=Optimization(batch_size=batch_size,
                        num_restarts_opti=num_restarts_opti,
                        device=device,
                        dtype=dtype,
                        raw_samples=raw_samples)

    niter=30
    num_replications = 5

    init_df=torch.load(f"./initial_samples/init_df_route_error_route_file_is{id_init_sample}.pt",weights_only=False)
    bo_loop=BOLoop(niter=niter,
                    init_df=init_df,
                    gp_initializer=gp_initializer,
                    objective_function=error_function,
                    optimizer=optimizer,
                    dtype=dtype,
                    x_dim=x_dim,
                    loss_scale=scaling_loss_factor,
                    output_dir=output_dir,
                    bounds=x_bounds)
    
    bo_loop.run_loop(kernel_name="linear_regression",
                        mean_name="linear_regression",
                        annotation=annotation,
                        rs=rs,
                        num_replications=num_replications)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kernel_name",
        type=str,
        required=True,
    )

    parser.add_argument(
            "--mean_name",
            type=str,
            required=True
    )


    parser.add_argument(
        "--experiment_id",
        type=int,
        required=True
    )

    args = parser.parse_args()

    id_init_samples = [0, 1, 2]
    restarts = [0, 1, 2]

    experiments = list(product(id_init_samples, restarts))

    if args.experiment_id >= len(experiments):
        parser.error("Invalid experiment_id")

    id_init_sample, rs = experiments[args.experiment_id]

    print(f"Experiment ID: {args.experiment_id}")
    print(f"Initial sample: {id_init_sample}")
    print(f"Restart: {rs}")


    run_bo(kernel_name=args.kernel_name,
           mean_name=args.mean_name,
           id_init_sample=id_init_sample,
           rs=rs)