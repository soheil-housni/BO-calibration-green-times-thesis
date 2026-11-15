import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


class ConnexionsGreenTime():
    def __init__(self,
                 tl_ids,
                 complete_corridors_connexions_df,
                 net_xml,
                 g_dict_to_change,
                 g_dict_to_vary,
                 g_dict_const,
                 dtype):
        
        self.tl_ids=tl_ids
        self.complete_corridors_connexions_df=complete_corridors_connexions_df.sort_values(by=["connexion_id"])
        self.net_xml=net_xml
        self.g_dict_to_change=g_dict_to_change
        self.g_dict_to_vary=g_dict_to_vary
        self.g_dict_const=g_dict_const
        self.dtype=dtype


    def extraction_connexions_green_time(self,
                                         x_init,
                                         index_to_vary,
                                         index_to_change,
                                         index_const,
                                         n_phases,
                                         tensor_matrix):
         
         g_vec_all=self.creation_g_vec_all(x_init,
                                            index_to_vary,
                                            index_to_change,
                                            index_const,
                                            n_phases)
         
         connexion_gt_vector=torch.matmul(g_vec_all,tensor_matrix.T)
         return connexion_gt_vector

    def fix_data_preparation(self):
        index_to_vary,index_to_change,index_const,n_phases,df_phases=self.extraction_indices()
        tensor_matrix,all_connexion_phases_df=self.creation_route_phase_correspondance_matrix(df_phases)
        return index_to_vary,index_to_change,index_const,n_phases,tensor_matrix,all_connexion_phases_df


    def creation_route_phase_correspondance_matrix(self,
                                                   df_phases):
         
         df_phases["tl_id_state"]=df_phases.apply(lambda x : (x["tl_id"],x["state"]), axis=1)
         phases=df_phases["tl_id_state"].values
         index = pd.MultiIndex.from_tuples(phases, names=["tl_id", "state"])
         connexion_index=np.sort(self.complete_corridors_connexions_df["connexion_id"].values)
         connexion_phases_df=pd.DataFrame(np.zeros((len(self.complete_corridors_connexions_df), len(phases))),columns=index,index=connexion_index)

         for i in connexion_index:
            tl=self.complete_corridors_connexions_df.loc[self.complete_corridors_connexions_df["connexion_id"]==i,"tl"].iloc[0]
            green_phases=self.complete_corridors_connexions_df.loc[self.complete_corridors_connexions_df["connexion_id"]==i,"green_phases"].iloc[0]
            for green_phase in green_phases:
                 connexion_phases_df.loc[i,(tl,green_phase)]=1

         connexion_phases_df.columns=connexion_phases_df.columns.to_flat_index()
         connexion_phases_df=connexion_phases_df.reset_index().rename(columns={"index":"connexion_id"})
         all_connexion_phases_df=pd.merge(self.complete_corridors_connexions_df,connexion_phases_df,on="connexion_id",how="inner")
         tensor_matrix=torch.tensor(all_connexion_phases_df[phases].values)
         return tensor_matrix,all_connexion_phases_df
                 


    def creation_g_vec_all(self,
                            x_init,
                            index_to_vary,
                            index_to_change,
                            index_const,
                            n_phases):
    
            g_vec_all=torch.zeros((*x_init.shape[:-1],n_phases),dtype=self.dtype)
    
            g_vec_delta=torch.tensor([90-15,90-15,90-6,90-15,90-15,90-15,90-39],dtype=self.dtype).unsqueeze(0)
            g_vec_to_change=g_vec_delta-x_init
    
            g_vec_const=torch.tensor([3,6,3,3,3,6,3,3,3,3,3,6,3,3,3,6,3,3,3,6,3,3,3,25,5,3,3],dtype=self.dtype).unsqueeze(0)
    
            g_vec_all[...,index_to_vary]=x_init
            g_vec_all[...,index_to_change]=g_vec_to_change
            g_vec_all[...,index_const]=g_vec_const
    
            return g_vec_all




    def extraction_indices(self,
                               ):
    
        tree = ET.parse(self.net_xml)
        root = tree.getroot()

        rows = []

        for tl in root.findall("tlLogic"):
            tl_id = tl.attrib["id"]
            
            for phase_idx, phase in enumerate(tl.findall("phase")):
                rows.append({
                    "tl_id": tl_id,
                    "phase_idx": phase_idx,
                    "state": phase.attrib["state"]
                })

        df_phases = pd.DataFrame(rows)


        df_phases["tl_id"] = pd.Categorical(
            df_phases["tl_id"],
            categories=self.tl_ids,
            ordered=True
        )

        df_phases = df_phases.sort_values(["tl_id","phase_idx"])
        df_phases=df_phases.reset_index().drop(columns="index").reset_index()


        df_to_vary=pd.DataFrame([(tl_id,state) for tl_id,state in self.g_dict_to_vary.items()],columns=["tl_id","state"])
        index_to_vary=pd.merge(df_phases,df_to_vary,how="inner",on=["tl_id","state"])["index"]

        df_to_change=pd.DataFrame([(tl_id,state) for tl_id,state in self.g_dict_to_change.items()],columns=["tl_id","state"])
        index_to_change=pd.merge(df_phases,df_to_change,how="inner",on=["tl_id","state"])["index"]

        df_const= pd.DataFrame([(tl_id, state) for tl_id, states in self.g_dict_const.items() for state in states], columns=["tl_id", "state"])
        index_const=pd.merge(df_phases,df_const,how="inner",on=["tl_id","state"])["index"]

        n_phases=len(df_phases)

        return index_to_vary,index_to_change,index_const,n_phases,df_phases