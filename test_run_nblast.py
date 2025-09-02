import json
import os
from unittest import TestCase

from run_nblast import nblast_list


class Test(TestCase):
    def test_batch_nblast(self):
        files = []
        for fname in sorted(os.listdir("swc")):
            if fname.endswith(".swc"):
                files.append(int(fname.split(".")[0]))
        print(f"Found {len(files)} swc files")
        files = files[:100] + [f for f in files if
                               f in [720575940632852124, 720575940626390018, 720575940627514627, 720575940621624491,
                                     720575940621089741, 720575940609617614, 720575940636592270, 720575940636734382,
                                     720575940638642547, 720575940609030776, 720575940627947772, 720575940619559838,
                                     720575940612033253, 720575940624710653]]
        print(f"Filtered {len(files)} swc files")
        res = nblast_list(files, min_score=0.1)
        print(json.dumps({str(k): v for k, v in res.items()}, indent=2))



