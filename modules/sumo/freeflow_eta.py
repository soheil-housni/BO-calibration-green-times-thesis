import numpy as np
import xml.etree.ElementTree as ET 
import pandas as pd

def get_freeflow_eta(gt_df,net_xml):
    net_df=pd.read_xml(net_xml,iterparse={"lane":{"id","speed","length"}}).rename(columns={"id":"lane_id","speed":"ff_speed"})
    net_df["edge_id"]=net_df["lane_id"].str.rsplit("_",n=1).str[0]
    net_df=net_df.groupby("edge_id").agg({"ff_speed":"mean","length":"mean"}).reset_index()
    gt_df=pd.merge(gt_df,net_df,how="left",on="edge_id")
    gt_df["ff_tt"]=gt_df["length"]/gt_df["ff_speed"]
    return gt_df
