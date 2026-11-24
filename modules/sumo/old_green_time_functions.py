import xml.etree.ElementTree as ET
import numpy as np

def update_green_times(g_vec,
                       additional_xml,
                       tl_ids_set):

    green_phase_times_dict = update_green_time_dict(g_vec)
    
    tree = ET.parse(additional_xml)
    root = tree.getroot()
    
    for tl in root.iter("tlLogic"):
        if tl.attrib["id"] in tl_ids_set:
            tl_curr = tl.attrib["id"]
            if tl.attrib["programID"] != "1":
                raise ValueError('We should only be changing the signal of programID 1')
            
            for ph in tl:
                if (tl_curr, ph.attrib["state"]) in green_phase_times_dict:                
                    ph.attrib["duration"] = str(int(green_phase_times_dict[tl_curr, ph.attrib["state"]]))
    tree.write(additional_xml)     


def update_green_time_dict(g_vec):

    g_vec_rounded = [int(np.floor(elem)) for elem in g_vec]

    g_all = map_to_all_green_times(g_vec_rounded)
    

    green_phase_new_times = {
        ("gneJ260", "GGGGGgrrr"): g_all[0],
        ("gneJ260", "GrrrrrGGG"): g_all[1],
        ("gneJ210", "GGggrrrrrrGGGG"): g_all[2],
        ("gneJ210", "rrrrGGGGGGGGrr"): g_all[3],
        ("32564122", "GGGGGgrrr"): g_all[4], 
        ("32564122", "GrrrrrGGG"): g_all[5], 
        ("gneJ207", "GGgGrGGG"): g_all[6],
        ("gneJ207", "rrrGGGrr"): g_all[7],
        ("gneJ143", "rrrGGGGgGGGg"): g_all[8],
        ("gneJ143", "GGGGrrrrrrrr"): g_all[9],
        ("cluster_1757124350_1757124352", "GGgrrGGG"): g_all[10],
        ("cluster_1757124350_1757124352", "rrrGGGrr"): g_all[11],
        ("cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190","GGGGGGrrrrrr"):g_all[12],
        ("cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190","rrrrrrrrGGGG"):g_all[13],
        }    


    return green_phase_new_times


def map_to_all_green_times(g_vec_rounded):
    g_all = [
        g_vec_rounded[0], 90-15-g_vec_rounded[0],
        g_vec_rounded[1], 90-15-g_vec_rounded[1],
        g_vec_rounded[2], 90-6-g_vec_rounded[2], #why 6 ? 
        g_vec_rounded[3], 90-15-g_vec_rounded[3],
        g_vec_rounded[4], 90-15-g_vec_rounded[4],
        g_vec_rounded[5], 90-15-g_vec_rounded[5],
        g_vec_rounded[6], 90-39-g_vec_rounded[6]         
    ]

    return g_all