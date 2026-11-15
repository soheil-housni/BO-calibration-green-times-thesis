import numpy as np
import sumolib
import pandas as pd
from modules.sumo import update_green_time_dict

class UpdatedFA():
    def __init__(self,
                 route_file,
                 net_file,
                 corridors_df,
                 constant_gt_dict):
        

        self.net=sumolib.net.readNet("./ingolstadt7/ingolstadt7.net.xml",withPrograms=True)
        self.routes_df=pd.read_xml("./ingolstadt7/ingolstadt7_routes_with_edges.rou.xml", iterparse={"route":["edges"]}).rename(columns={"edges":"route_edges"})
        self.corridors_df=corridors_df
        self.constant_gt_dict=constant_gt_dict

    def fA(self,g_vec,df_gt):
        traveltime_connexion_corridors_df=self.predicted_traveltime(g_vec)
        traveltime_connexion_corridors_df=traveltime_connexion_corridors_df.drop_duplicates(["tl","route_str"])
        gt_tt_dict=df_gt.set_index("edge_id")["average_traveltime"].to_dict()
        traveltime_connexion_corridors_df["route_gt_traveltime"]=traveltime_connexion_corridors_df["route"].apply(self.extract_route_traveltime,args=(gt_tt_dict,))
        traveltime_connexion_corridors_df["squared_error"]=np.square(traveltime_connexion_corridors_df["predicted_route_sum_average_traveltime"]-traveltime_connexion_corridors_df["route_gt_traveltime"])
        return traveltime_connexion_corridors_df["squared_error"].mean()

    def extract_route_traveltime(self,route,tt_dict):
        route_traveltime=[]
        for edge in route:
            route_traveltime.append(tt_dict[edge])
        return np.sum(route_traveltime)


    def predicted_traveltime(self,g_vec):

        connexion_corridors_df=self.create_connexion_corridors_df()


        gt_dict=update_green_time_dict(g_vec)
        
        gt_dict_updated=gt_dict.copy()

        gt_dict_updated.update(self.constant_gt_dict)

        gt_serie=pd.Series(gt_dict_updated)

        connexion_corridors_df["connexion_gt_duration"]=connexion_corridors_df.apply(self.extract_gt_connexion,args=(gt_serie,),axis=1)

        agg_metrics_df=connexion_corridors_df.groupby(["tl","route_str","route_last_edge","to_edge"]).agg({"connexion_demand":"mean","connexion_gt_duration":"mean"}).rename(columns={"connexion_demand":"average_connexion_demand","connexion_gt_duration":"average_connexion_gt_duration"}).reset_index()

        connexion_corridors_df=pd.merge(connexion_corridors_df,agg_metrics_df,how="left",on=['tl', 'route_str', 'route_last_edge', 'to_edge']).drop_duplicates(["tl","route_str","route_last_edge","to_edge"])

        connexion_corridors_df["gt"]=connexion_corridors_df["average_connexion_gt_duration"]

        connexion_corridors_df["cycle_time"]=90
        connexion_corridors_df["proport_ff_tt"]=(connexion_corridors_df["gt"]/connexion_corridors_df["cycle_time"])*connexion_corridors_df["route_sum_ff_tt"]
        connexion_corridors_df["inverse_proport_ff_tt"]=1/connexion_corridors_df["proport_ff_tt"]
        connexion_corridors_df["normalized_gt"]=connexion_corridors_df["gt"]/connexion_corridors_df["route_sum_ff_tt"]
        connexion_corridors_df["normalized_critical_gt"]=(27.79340206/(connexion_corridors_df["route_sum_ff_tt"]-0.10356882)) + 0.10519363
        connexion_corridors_df["normalized_critical_gt_diff"]=connexion_corridors_df["normalized_gt"]-connexion_corridors_df["normalized_critical_gt"]
        connexion_corridors_df["critical_ff_speed_length_product"]=connexion_corridors_df["normalized_critical_gt_diff"]*connexion_corridors_df["route_sum_length"]*connexion_corridors_df["route_ff_speed"]
        connexion_corridors_df["square_inverse_proport_ff_tt"]=np.square(connexion_corridors_df["inverse_proport_ff_tt"])
        predictors=connexion_corridors_df[["inverse_proport_ff_tt",
                            "normalized_gt",
                            "normalized_critical_gt_diff", 
                            "route_sum_ff_tt",
                            "route_ff_speed",
                            "critical_ff_speed_length_product",
                            "square_inverse_proport_ff_tt"]].values
            
        connexion_corridors_df["predicted_connexion_delay_factor"]=self.linear_regression_model(predictors)
        connexion_corridors_df["predicted_connexion_route_sum_average_traveltime"]=connexion_corridors_df["predicted_connexion_delay_factor"]*connexion_corridors_df["route_sum_ff_tt"]
        connexion_corridors_df["predicted_route_sum_average_traveltime"]=connexion_corridors_df.apply(self.extract_average_predicted_traveltime,args=(connexion_corridors_df,),axis=1)

        return connexion_corridors_df
    
    def linear_regression_model(self,predictors):
        
        b0=-6.40621988e+00
        b1=8.57352254e+01
        b2=-7.41941697e+00
        b3=1.10365529e+01
        b4=1.86797158e-01
        b5=4.04908884e-01
        b6=-2.79010501e-04
        b7=-1.20503437e+01

        coeff=np.array([b0,b1,b2,b3,b4,b5,b6,b7])

        result=coeff[0]+np.dot(coeff[1:],predictors.T)
        result[result<0]=0

        return result


    def extract_average_predicted_traveltime(self,row,corridors_df):
        route_str=row["route_str"]
        route_filtered_df=corridors_df.loc[corridors_df["route_str"]==route_str]
        predicted_route_sum_average_traveltime=(route_filtered_df["predicted_connexion_route_sum_average_traveltime"]*route_filtered_df["average_connexion_demand"]).sum()/route_filtered_df["average_connexion_demand"].sum()
        return predicted_route_sum_average_traveltime 

    def extract_gt_connexion(self,row,gt_serie):
        tl=row["tl"]
        green_phases=row["green_phases"]
        gt_duration=0
        for phase in green_phases:
            gt_duration+=gt_serie.loc[tl,phase]
        return gt_duration

    def create_connexion_corridors_df(self):
        connections_df=self.creation_connexion_df()
        corridors_df_merged=pd.merge(self.corridors_df,connections_df,how="left",on="route_last_edge")
        corridors_df_merged["green_phases"]=corridors_df_merged.apply(self.extract_green_phases,args=(self.net,),axis=1)
        connexions_demand_df=self.extract_connexion_demand(corridors_df_merged,self.routes_df)
        connexion_corridors_df=pd.merge(corridors_df_merged,connexions_demand_df,how="left",on=["tl","route_str","route_last_edge","to_edge","link_index"])
        return connexion_corridors_df


    def extract_connexion_demand(self,corridors_df,routes_df):
        connexions_demand_dict=[]
        for index,row in corridors_df[["tl","route_str","route_last_edge","to_edge","link_index"]].drop_duplicates().iterrows():
            connexion_demand=routes_df["route_edges"].str.contains(f"{row['route_last_edge']} {row['to_edge']}").sum()
            connexions_demand_dict.append({
                "tl":row["tl"],
                "route_str":row["route_str"],
                "route_last_edge":row["route_last_edge"],
                "to_edge":row["to_edge"],
                "link_index":row["link_index"],
                "connexion_demand":connexion_demand,
            })

        connexions_demand_df=pd.DataFrame(connexions_demand_dict)
        return connexions_demand_df

    def extract_green_phases(self,row,net):
        from_edge=row["route_last_edge"]
        to_edge=row["to_edge"]
        tl=row["tl"]
        linkIndex=row["link_index"]
        traffic_light = net.getTLS(tl)
        green_phases=[]
        for program_id, program in traffic_light.getPrograms().items():
            for i, phase in enumerate(program.getPhases()):
                phase_state=phase.state
                linkIndexPhaseElem=phase_state[linkIndex]
                if (linkIndexPhaseElem=="G") or (linkIndexPhaseElem=="g"):
                    green_phases.append(phase_state)

        return green_phases


    def creation_connexion_df(self):
        last_edges_list=[]
        to_edges_list=[]
        link_index_list=[]

        for edge_id in self.corridors_df["route_last_edge"].unique():
            edge = self.net.getEdge(edge_id)

            to_edges = edge.getOutgoing()

            for to_edge in to_edges:
                to_edge_id = to_edge.getID()

                connections=edge.getConnections(to_edge)
                for connection in connections:
                    if connection.getFrom().getID()==edge_id:
                        link_index = connection.getTLLinkIndex()

                        last_edges_list.append(edge_id)
                        to_edges_list.append(to_edge_id)
                        link_index_list.append(link_index)

        connections_df=pd.DataFrame({"route_last_edge":last_edges_list,"to_edge":to_edges_list,"link_index":link_index_list})
        return connections_df