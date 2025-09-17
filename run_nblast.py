import json
import os
import sys
from pathlib import Path

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r

from instrumentation import TimingLogger

r("options(rgl.useNULL=TRUE)")
nat = importr("nat")
nblast = importr("nat.nblast")

tlog = TimingLogger("NBLAST/py")

def load_dp(rid, folder):
    dct = load_dps([rid], folder)
    return dct[rid]

def load_dps(ids, folder):
    tlog.report(f"Loading DPs for {len(ids)} IDs")
    base_dir = os.path.join(os.getcwd(), folder)

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


def nblast_file_path_pair(file1, file2):
    path1, path2 = Path(file1), Path(file2)
    return nblast_id_pair(path1.stem, path2.stem, str(path1.parent), str(path2.parent))


def nblast_id_pair(id1: str, id2: str, folder1: str, folder2: str) -> float:
    return nblast_dp_pair(load_dp(id1, folder1), load_dp(id2, folder2))


def nblast_dp_pair(dp1, dp2) -> float:
    score1 = nblast.nblast(dp1, dp2, normalised=True)
    score2 = nblast.nblast(dp2, dp1, normalised=True)
    return (score1[0] + score2[0]) / 2


def nblast_all_by_all(ids, folder, min_score) -> dict:
    dps_dict = load_dps(ids, folder)
    tlog.report(f"Running all by all nblast for {len(dps_dict)} SWCs")
    res_vec = nblast.nblast_allbyall(nat.as_neuronlist(ro.ListVector(dps_dict)), normalisation="mean")
    tlog.report(f"Storing scores in dict {len(dps_dict)} SWCs")
    dimnames = res_vec.do_slot("dimnames")
    rownames = list(dimnames[0])
    colnames = list(dimnames[1])
    res_dict = {
        (rrid, crid): res_vec[i * len(ids) + j]
        for i, rrid in enumerate(rownames)
        for j, crid in enumerate(colnames)
    }
    res_dict = {k: v for k, v in res_dict.items() if v >= min_score}
    tlog.report(f"[nblast_all_by_all] Stored {len(res_dict)} scores >{min_score}")
    return res_dict


def nblast_list_to_list(ids1, ids2, folder1, folder2, min_score) -> dict:
    dps_dict_1 = load_dps(ids1, folder1)
    dps_dict_2 = load_dps(ids2, folder2)
    tlog.report(f"Running list to list nblast for {len(dps_dict_1)} X {len(dps_dict_2)} SWCs")
    res_dict = {}
    for f1, dp1 in dps_dict_1.items():
        for f2, dp2 in dps_dict_2.items():
            sc = float(nblast_dp_pair(dp1, dp2))
            if sc >= min_score:
                res_dict[(f1, f2)] = sc
                if (f2, f1) in res_dict:
                    assert sc == res_dict[(f2, f1)]
                else:
                    res_dict[(f2, f1)] = sc
    tlog.report(f"[nblast_list_to_list] Stored {len(res_dict)} scores >{min_score}")
    return res_dict

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_nblast.py neuron1.swc neuron2.swc")
        sys.exit(1)
    file1, file2 = sys.argv[1], sys.argv[2]
    try:
        score = nblast_file_path_pair(file1, file2)
        print(f"NBLAST score between {file1} and {file2}: {score}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
