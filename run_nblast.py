import json
import os
import sys
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r
r("options(rgl.useNULL=TRUE)")
nat = importr("nat")
nblast = importr("nat.nblast")


def nblast_pair(file1: str, file2: str) -> float:
    base_dir = os.path.join(os.getcwd(), "swc")
    path1 = os.path.join(base_dir, file1)
    path2 = os.path.join(base_dir, file2)

    if not os.path.exists(path1):
        raise FileNotFoundError(f"SWC file not found: {path1}")
    if not os.path.exists(path2):
        raise FileNotFoundError(f"SWC file not found: {path2}")

    # Load neurons
    neuron1 = r['read.neuron'](path1)
    neuron2 = r['read.neuron'](path2)

    # Convert to dotprops
    dot1 = nat.dotprops(neuron1)
    dot2 = nat.dotprops(neuron2)

    # Run NBLAST
    score = nblast.nblast(dot1, dot2, normalised=True)
    return float(score[0])


def nblast_list(files) -> dict:
    base_dir = os.path.join(os.getcwd(), "swc")
    paths = [os.path.join(base_dir, f"{f}.swc") for f in files]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"SWC file not found: {p}")

    neurons = [r['read.neuron'](p) for p in paths]
    dps = [nat.dotprops(n) for n in neurons]
    r_list = ro.ListVector({str(files[i]): dps[i] for i in range(len(files))})
    res_vec = nblast.nblast_allbyall(nat.as_neuronlist(r_list), normalisation="mean")
    res_dict = {}
    for i, f1 in enumerate(files):
        for j, f2 in enumerate(files):
            score = res_vec[len(files) * i + j]
            res_dict[(f1, f2)] = score
            assert res_dict.get((f2, f1), score) == score
    return res_dict


def main():
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


if __name__ == "__main__":
    # main()

    files = []
    for fname in sorted(os.listdir("swc")):
        if fname.endswith(".swc"):
            files.append(int(fname.split(".")[0]))
    print(f"Found {len(files)} swc files")
    files = [f for f in files if f in [720575940632852124,720575940626390018,720575940627514627,720575940621624491,720575940621089741,720575940609617614,720575940636592270,720575940636734382,720575940638642547,720575940609030776,720575940627947772,720575940619559838,720575940612033253,720575940624710653]]
    print(f"Filtered {len(files)} swc files")
    res = nblast_list(files)
    print(json.dumps({str(k): v for k, v in res.items()}, indent=2))



