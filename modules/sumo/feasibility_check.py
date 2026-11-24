from .old_green_time_functions import map_to_all_green_times
import numpy as np

def check_feasibility(g_vec):
        # round down to nearest integer
        g_vec_rounded = [int(np.floor(elem)) for elem in g_vec]

        # infer the green times of the other signal phases that depend on the decision variables
        g_all = map_to_all_green_times(g_vec_rounded)

        print("G_ALL :")
        print(g_all)

        # boolean indicating whether any green phase is non-positive
        return len(np.where(np.array(g_all)<=0)[0])==0