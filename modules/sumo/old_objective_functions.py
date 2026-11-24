import numpy as np
import pandas as pd
from .old_green_time_functions import update_green_times
import shutil
from .vehroute_df_preprocessing import create_vehroute_df
from .extract_route_connexion_traveltime import get_route_connexion_average_traveltime

class ErrorFunctionRouteVersion():
    def __init__(self,
                 scaling_factor_loss_fn,
                 tl_ids,
                 df_gt,
                 sumo_simulator,
                 corridors_df,
                 additional_xml,
                 output_dir):
        
        self.scaling_factor_loss_fn=scaling_factor_loss_fn
        self.df_gt=df_gt
        self.sumo_simulator=sumo_simulator
        self.additional_xml=additional_xml
        self.output_dir=output_dir

        self.tl_ids=tl_ids
        self.corridors_df=corridors_df.sort_values(by="route_last_edge")
        
        self.corridors_df_gt=self.corridors_df.copy()
        
        self.tt_dict_gt = self.df_gt.set_index("edge_id")["traveltime_mean"].to_dict()
        
        self.corridors_df_gt["route_sum_average_traveltime"]=self.corridors_df_gt["route"].apply(self.extract_route_traveltime,args=(self.tt_dict_gt,))
        


    def compute_error(self,
                      g_vec,
                      epoch_index,
                      num_replications,
                      travel_time=False,
                      demand_scaling_factor=None):

        update_green_times(g_vec=g_vec,additional_xml=self.additional_xml,tl_ids_set=self.tl_ids)
        shutil.copy(self.additional_xml, f'{self.output_dir}{epoch_index}_copy_additional.xml')
        mse_list=[]
        tt_list=[]

        for rep in range(num_replications):
            self.sumo_simulator.simulation(epoch_index=epoch_index, rep=rep,demand_scaling_factor=demand_scaling_factor)
            df_sim=pd.read_csv(f"{self.output_dir}{epoch_index}_{rep}_InTAS7DIM.travel_times.csv")
            tt_dict_sim = df_sim.set_index("edge_id")["traveltime"].to_dict()
            corridors_df_sim=self.corridors_df.copy()
            corridors_df_sim["verif"]=corridors_df_sim["route"].apply(self.filtering,args=(df_sim,))
            corridors_df_sim_filtered=corridors_df_sim[corridors_df_sim["verif"]]
            corridors_df_sim_filtered["route_sum_average_traveltime"]=corridors_df_sim_filtered["route"].apply(self.extract_route_traveltime,args=(tt_dict_sim,))
            corridors_df_gt_filtered=self.corridors_df_gt[corridors_df_sim["verif"]]
            mse=np.square((corridors_df_sim_filtered["route_sum_average_traveltime"]/self.scaling_factor_loss_fn)-(corridors_df_gt_filtered["route_sum_average_traveltime"]/self.scaling_factor_loss_fn)).mean()
            mse_list.append(mse)
            tt_list.append(corridors_df_sim["route_sum_average_traveltime"].mean())

        if travel_time:
            return (np.mean(mse_list),np.mean(tt_list))
        else:
            return np.mean(mse_list) 

    def extract_route_traveltime(self,route,tt_dict):
        route_traveltime=[]
        for edge in route:
            route_traveltime.append(tt_dict[edge])
        return np.sum(route_traveltime)

    def filtering(self,route,df_sim):
        verif=True
        for edge in route:
            if edge not in df_sim["edge_id"].values:
                verif=False
        return verif







class ErrorFunctionConnexionVersion():
    def __init__(self,
                 scaling_factor_loss_fn,
                 df_gt,
                 sumo_simulator,
                 complete_corridors_connexions_df,
                 additional_xml,
                 output_dir,
                 tl_ids):
        
        self.scaling_factor_loss_fn=scaling_factor_loss_fn
        self.sumo_simulator=sumo_simulator
        self.additional_xml=additional_xml
        self.output_dir=output_dir
        self.tl_ids=tl_ids


        self.complete_corridors_connexions_df=complete_corridors_connexions_df.sort_values(by="connexion_id")
        self.df_gt=df_gt=df_gt.sort_values(by="connexion_id")
        


    def compute_error(self,
                      g_vec,
                      epoch_index,
                      num_replications,
                      travel_time=False,
                      demand_scaling_factor=None):

        update_green_times(g_vec=g_vec,additional_xml=self.additional_xml,tl_ids_set=self.tl_ids)
        shutil.copy(self.additional_xml, f'{self.output_dir}{epoch_index}_copy_additional.xml')
        mse_list=[]
        tt_list=[]

        for rep in range(num_replications):

            self.sumo_simulator.simulation(epoch_index=epoch_index, rep=rep,demand_scaling_factor=demand_scaling_factor)
            vehroute_df=create_vehroute_df(f"ingolstadt7/{self.output_dir}{epoch_index}_{rep}_InTAS7DIM.vehroute.xml")
            complete_corridors_connexions_df_sim=self.complete_corridors_connexions_df.copy()
            complete_corridors_connexions_df_sim["route_connexion_average_traveltime"]=complete_corridors_connexions_df_sim.apply(get_route_connexion_average_traveltime,args=(vehroute_df,),axis=1)
            merged_df=pd.merge(self.df_gt[["connexion_id","route_connexion_average_traveltime"]],complete_corridors_connexions_df_sim[["connexion_id","route_connexion_average_traveltime"]],suffixes=["_gt","_sim"],on="connexion_id",how="inner")
            merged_df=merged_df.dropna(subset=["route_connexion_average_traveltime_sim","route_connexion_average_traveltime_gt"],axis=0, how="any")

            mse=np.square(
                merged_df["route_connexion_average_traveltime_sim"]-merged_df["route_connexion_average_traveltime_gt"]
            ).mean()

            average_tt=merged_df["route_connexion_average_traveltime_sim"].mean()

            mse_list.append(mse)
            tt_list.append(average_tt)

        if travel_time:
            return (np.mean(mse_list),np.mean(tt_list))
        else:
            return np.mean(mse_list) 
