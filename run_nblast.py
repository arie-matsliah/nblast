import json
import os
import sys
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r
r("options(rgl.useNULL=TRUE)")
nat = importr("nat")
nblast = importr("nat.nblast")


def load_dps(ids):
    base_dir = os.path.join(os.getcwd(), "swc")

    def to_swc_fname(f):
        if isinstance(f, str) and f.endswith(".swc"):
            return f
        return f"{f}.swc"

    paths = [os.path.join(base_dir, to_swc_fname(f)) for f in ids]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"SWC file not found: {p}")

    dps_dict = {}
    for f, p in zip(ids, paths):
        try:
            n = r['read.neuron'](p)
            dps_dict[str(f)] = nat.dotprops(n)
        except Exception as e:
            print(f"Exception for {f}: {e}")
    return dps_dict


def nblast_pair(file1: str, file2: str) -> float:
    dps_dict = load_dps([file1, file2])
    score = nblast.nblast(dps_dict[file1], dps_dict[file2], normalised=True)
    return float(score[0])


def nblast_list(ids, min_score) -> dict:
    dps_dict = load_dps(ids)
    res_vec = nblast.nblast_allbyall(nat.as_neuronlist(ro.ListVector(dps_dict)), normalisation="mean")
    res_dict = {}
    flist = sorted(dps_dict.keys())
    for i, f1 in enumerate(flist):
        for j, f2 in enumerate(flist):
            score = res_vec[len(flist) * i + j]
            if score >= min_score:
                res_dict[(f1, f2)] = score
                assert res_dict.get((f2, f1), score) == score
    return res_dict


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_nblast.py neuron1.swc neuron2.swc")
        sys.exit(1)
    file1, file2 = sys.argv[1], sys.argv[2]
    try:
        score = nblast_pair(file1, file2)
        print(f"NBLAST score between {file1} and {file2}: {score}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
