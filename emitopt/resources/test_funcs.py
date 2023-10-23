# +
from emitopt.beam_dynamics import twiss_transport_mat_from_rmat, build_quad_rmat

def single_quadrupole_with_scaling(k, s):
    quad_rmat = 


def eval_beamsize(input_dict):
    k = input_dict["x0"]
    s = torch.tensor([input_data[f"x{i}"] for i in range(1, len(input_dict.keys()))])
    
# -


