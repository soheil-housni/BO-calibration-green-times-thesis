import numpy as np
import pandas as pd
import torch

def connexions_routes_correspondance_matrix(complete_corridors_connexions_df):
    routes_set=np.sort(complete_corridors_connexions_df["route_str_str"].unique())
    connexions_set=complete_corridors_connexions_df["connexion_id"].unique()
    connexions_routes_correspondance_df=pd.DataFrame(np.zeros((len(connexions_set),len(routes_set))),index=connexions_set,columns=routes_set)
    for i in connexions_set:
        route_str_str=complete_corridors_connexions_df.loc[complete_corridors_connexions_df["connexion_id"]==i,"route_str_str"].iloc[0]
        connexions_routes_correspondance_df.loc[i,route_str_str]=1
    connexions_routes_correspondance_tensor=torch.tensor(connexions_routes_correspondance_df.values)
    return connexions_routes_correspondance_tensor,connexions_routes_correspondance_df