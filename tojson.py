import json
import numpy as np
from score import compute_dist_mat,load_instance

inst1 = "data/inst1"
inst2 = "data/inst2"
inst3 = "data/inst3"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)



distmat={}

distmat["inst1"]=compute_dist_mat(load_instance(inst1))
distmat["inst2"]=compute_dist_mat(load_instance(inst2))
distmat["inst3"]=compute_dist_mat(load_instance(inst3))


with open ("distmat.json","w") as file:
    json.dump(distmat,file,cls=NumpyEncoder)