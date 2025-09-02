import json
import os
import sys
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r

from instrumentation import TimingLogger

r("options(rgl.useNULL=TRUE)")
nat = importr("nat")
nblast = importr("nat.nblast")

tlog = TimingLogger("NBLAST/py")


def load_dps(ids):
    tlog.report(f"Loading DPs for {len(ids)} IDs")
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
            tlog.report(f"Exception for {f}: {e}")

    tlog.report(f"Loaded {len(dps_dict)} DPs out of {len(ids)}")
    return dps_dict


def nblast_pair(file1: str, file2: str) -> float:
    dps_dict = load_dps([file1, file2])
    score = nblast.nblast(dps_dict[file1], dps_dict[file2], normalised=True)
    return float(score[0])


def nblast_all_by_all(ids, min_score) -> dict:
    dps_dict = load_dps(ids)
    tlog.report(f"Running all by all nblast for {len(dps_dict)} SWCs")
    res_vec = nblast.nblast_allbyall(nat.as_neuronlist(ro.ListVector(dps_dict)), normalisation="mean")
    tlog.report(f"Storing scores in dict {len(dps_dict)} SWCs")
    res_dict = {}
    flist = sorted(dps_dict.keys())
    for i, f1 in enumerate(flist):
        for j, f2 in enumerate(flist):
            score = res_vec[len(flist) * i + j]
            if f1 != f2 and score >= min_score:
                res_dict[(f1, f2)] = score
                assert res_dict.get((f2, f1), score) == score
    tlog.report(f"Stored {len(res_dict)} scores >{min_score}")
    return res_dict

def nblast_list_to_list(ids1, ids2, min_score) -> dict:
    dps_dict_1 = load_dps(ids1)
    dps_dict_2 = load_dps(ids2)
    tlog.report(f"Running list to list nblast for {len(dps_dict_1)} X {len(dps_dict_2)} SWCs")
    res_dict = {}
    for f1, dp1 in dps_dict_1.items():
        for f2, dp2 in dps_dict_2.items():
            if f1 == f2:
                continue
            s12 = nblast.nblast(dp1, dp2, normalised=True)
            s21 = nblast.nblast(dp2, dp1, normalised=True)
            score = (float(s12[0]) + float(s21[0])) / 2
            if score >= min_score:
                res_dict[(f1, f2)] = score
                res_dict[(f2, f1)] = score
    tlog.report(f"Stored {len(res_dict)} scores >{min_score}")
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
