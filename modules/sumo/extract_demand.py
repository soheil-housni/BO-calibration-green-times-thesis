import pandas as pd

def extract_route_demand(routes_file,corridors_df):
    routes_demand_dict={}
    routes_df=pd.read_xml(routes_file, iterparse={"route":["edges"]}).rename(columns={"edges":"route_edges"})
    for index,row in corridors_df[["route_str","route_str_str"]].drop_duplicates().iterrows():
        route_demand=routes_df["route_edges"].str.contains(row["route_str_str"]).sum()
        routes_demand_dict[row["route_str"]]=route_demand
    return routes_demand_dict