import torch
import pandas as pd
import numpy as np
from modules.sumo import ErrorFunctionRouteVersion,TLIncomingCorridors,update_green_times,SumoSimulation,create_initial_tl_df
from modules.bo import InitialSamples
import ast
import argparse
import os
import shutil
import random
from lxml import etree



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Pour favoriser la reproductibilité
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_sumo_files(id_init_sample):

    network_dir = "./ingolstadt7"

    original_additional = os.path.join(
        network_dir, "additional.xml"
    )

    original_config = os.path.join(
        network_dir, "ingolstadt7.sumocfg"
    )

    # Fichiers propres à cet initial sample
    additional_xml = os.path.join(
        network_dir, f"additional_init_{id_init_sample}.xml"
    )

    cfg_file = os.path.join(
        network_dir, f"ingolstadt7_init_{id_init_sample}.sumocfg"
    )

    # Copier les fichiers originaux
    shutil.copy2(original_additional, additional_xml)
    shutil.copy2(original_config, cfg_file)

    # Modifier la configuration pour référencer le XML spécifique
    tree = etree.parse(cfg_file)

    additional_files = tree.find(".//additional-files")

    if additional_files is None:
        raise ValueError("additional-files introuvable dans le sumocfg")

    additional_files.set("value", f"additional_init_{id_init_sample}.xml")

    tree.write(cfg_file, encoding="UTF-8", xml_declaration=True)

    return cfg_file, additional_xml


def generate_initial_samples(id_init_sample):
    set_seed(id_init_sample)

    cfg_file, additional_xml = prepare_sumo_files(id_init_sample)

    x_dim=7
    init_output_dir="initial_output/"

    scaling_loss_factor=1

    num_replications = 5

    gt_additional_xml="ingolstadt7/additional.gt_scenario.xml"
    initial_tl_df=create_initial_tl_df(gt_additional_xml)

    tl_ids = np.sort(initial_tl_df["tl"].unique())

    df_gt=pd.read_csv("gt_data/df_gt_InTAS7DIM.csv")
    df_gt["route"]=df_gt["route"].apply(lambda x : ast.literal_eval(x))
    df_gt=df_gt.sort_values(by="route_id")


    low_bound = 5
    up_bound = 74
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    x_bounds = torch.tensor([[low_bound]*x_dim,
                        [up_bound]*(x_dim-1)+[50]], 
                        device=device, dtype=dtype)

    sumo_simulator=SumoSimulation(
        cfg_file=cfg_file,
        output_dir=init_output_dir,
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
                                             output_dir=init_output_dir,
                                             tl_ids=tl_ids)

    initial_sampler=InitialSamples(
        objective_function=error_function,
        dtype=dtype
    )


    init_sample_size=5

    
    init_sample_x_norm, init_sample_y, init_df = initial_sampler.get_initial_sample(sample_size=init_sample_size,
                                                                    x_dim=x_dim,
                                                                    x_bounds=x_bounds,
                                                                    seed=id_init_sample,
                                                                    id_init_sample=id_init_sample,
                                                                    num_replications=num_replications)

    torch.save(init_sample_x_norm,f"./initial_samples/init_sample_x_norm_route_error_route_file_is{id_init_sample}.pt")
    torch.save(init_sample_y,f"./initial_samples/init_sample_y_route_error_route_file_is{id_init_sample}.pt")
    torch.save(init_df,f"./initial_samples/init_df_route_error_route_file_is{id_init_sample}.pt")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
                "--id_init_sample",
                type=int,
                required=False
    )


    args = parser.parse_args()


    generate_initial_samples(id_init_sample=args.id_init_sample)