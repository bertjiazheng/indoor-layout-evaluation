#!/usr/bin/env python
import os
import sys
import time

import numpy as np

from metrics import evaluate

THRES = {'junction': [5, 10, 15], 'wireframe': [5, 10, 15], 'plane': 0.5}


def main():
    submit_dir = sys.argv[1]
    gt_dir = sys.argv[2]

    time_start = time.time()

    filelist = [filename for filename in sorted(
        os.listdir(gt_dir)) if filename.endswith('txt')]

    # check complete submission
    for filename in filelist:
        if not os.path.exists(os.path.join(submit_dir, filename)):
            sys.exit('Could not find submission file {0}'.format(filename))

    # compute final results
    results = np.zeros((len(filelist), 3))
    for index, filename in enumerate(sorted(filelist)):
        preds = np.loadtxt(os.path.join(submit_dir, filename), dtype=np.int)
        gts = np.loadtxt(os.path.join(gt_dir, filename), dtype=np.int)
        Fs = evaluate(gts, preds, THRES)
        results[index] = np.array(
            (np.mean(Fs['junction']), np.mean(Fs['wireframe']), np.mean(Fs['plane'])))

    total_time = time.time() - time_start

    print(f"F (Mean):\t{np.mean(results):.4f}\n"
          f"F (Junction):\t{np.mean(results[:, 0]):.4f}\n"
          f"F (Wireframe):\t{np.mean(results[:, 1]):.4f}\n"
          f"F (Plane):\t{np.mean(results[:, 2]):.4f}\n"
          f"\nTotal time:\t{total_time:.4f} (s)")


if __name__ == '__main__':
    main()
