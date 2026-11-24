from modules.sumo import TLIncomingCorridors,CompleteCorridorsConnexions
import torch

net_xml="./ingolstadt7/ingolstadt7.net.xml"
route_file="./ingolstadt7/ingolstadt7_routes_with_edges.rou.xml"
corridors_generator=TLIncomingCorridors(net_file=net_xml,route_file=route_file)
corridors_df=corridors_generator.get_corridors_df()
corridors_df=corridors_df.loc[(corridors_df["origin"]!="other_type") & (corridors_df["route_demand"]>0)]
torch.save(corridors_df,"./corridors_savings/corridors_df.pt")