from modules.sumo import ErrorFunctionConnexionVersion,TLIncomingCorridors,CompleteCorridorsConnexions,ConnexionsGreenTime,connexions_routes_correspondance_matrix,update_green_time_dict,update_green_times,SumoSimulation
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

    df_gt=pd.read_csv("./gt_data/df_gt_InTAS7DIM.csv")

    df_gt["route"]=df_gt["route"].apply(lambda x : ast.literal_eval(x))
    df_gt["green_phases"]=df_gt["green_phases"].apply(lambda x : ast.literal_eval(x))


    scaling_loss_factor=1

    def get_tls():
        green_phase_times_dict_tmp = update_green_time_dict(33*np.ones(x_dim))
        tl_ids_set = []
        for elem in green_phase_times_dict_tmp.keys():
            if elem[0] not in tl_ids_set:
                tl_ids_set.append(elem[0])
        return tl_ids_set   

    tl_ids = get_tls()
    print(tl_ids)


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

    g_dict_to_vary={
            "gneJ260":"GGGGGgrrr",
            "gneJ210": "GGggrrrrrrGGGG",
            "32564122": "GGGGGgrrr",
            "gneJ207":"GGgGrGGG",
            "gneJ143": "rrrGGGGgGGGg",
            "cluster_1757124350_1757124352":"GGgrrGGG",
            "cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190":"GGGGGGrrrrrr"
        }

    g_dict_to_change={
            "gneJ260":"GrrrrrGGG",
            "gneJ210": "rrrrGGGGGGGGrr",
            "32564122": "GrrrrrGGG",
            "gneJ207":"rrrGGGrr",
            "gneJ143": "GGGGrrrrrrrr",
            "cluster_1757124350_1757124352":"rrrGGGrr",
            "cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190":"rrrrrrrrGGGG"
        }

    g_dict_const={
        "gneJ260":["yyyyygrrr","rrrGGGrrr","rrryyyrrr","yrrrrryyy"],
        "gneJ210": ["yyggrrrrrryyyy","GGGGrrrrrrrrrr","yyyyrrrrrrrrrr","rrrryyyyyyyyrr"],
        "32564122": ["yyyyyyrrr","yrrrrryyy"],
        "gneJ207":["yygyryyy","GGGrrrrr","yyyrrrrr","rrryyyrr"],
        "gneJ143": ["rrryyyygyyyg","rrrrrrrGrrrG","rrrrrrryrrry","yyyyrrrrrrrr"],
        "cluster_1757124350_1757124352":["yygrryyy","GGGrrrrr","yyyrrrrr","rrryyyrr"],
        "cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190":["rrrrrrrrGGyy","rrrrrrGGGGrr","rrrrGGGGGGrr","rrrrGGyyyyrr","yyyyyyrrrrrr"]
    }


    complete_corridors_connexions_generator=CompleteCorridorsConnexions(route_xml,net_xml,corridors_df)

    complete_corridors_connexions_df=complete_corridors_connexions_generator.create_connexion_corridors_df()



    error_function=ErrorFunctionConnexionVersion(scaling_factor_loss_fn=scaling_loss_factor,
                                            df_gt=df_gt,
                                            sumo_simulator=sumo_simulator,
                                            complete_corridors_connexions_df=complete_corridors_connexions_df,
                                            additional_xml=additional_xml,
                                            output_dir=output_dir,
                                            tl_ids=tl_ids)




    connexions_green_time_extractor=ConnexionsGreenTime(
        tl_ids=tl_ids,
        complete_corridors_connexions_df=df_gt,
        net_xml=net_xml,
        g_dict_to_change=g_dict_to_change,
        g_dict_to_vary=g_dict_to_vary,
        g_dict_const=g_dict_const,
        dtype=dtype
    )

    fA_function=UpdatedFAFunctionLinearRegresion(
        df_gt=df_gt,
        connexions_green_time_extractor=connexions_green_time_extractor,
        dtype=dtype,
    )


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