#!/usr/bin/env python
import os
import sys
import time

import numpy as np

from metrics import evaluate

HEIGHT, WIDTH = 512, 1024
THRES = {'junction': [5, 10, 15], 'wireframe': [5, 10, 15], 'plane': 0.5}


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    time_start = time.time()

    # unzipped submission data is always in the 'res' subdirectory
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    submit_dir = os.path.join(input_dir, 'res')
    if not os.path.exists(submit_dir):
        sys.exit('Could not find submission file {0}'.format(submit_dir))

    # unzipped reference data is always in the 'ref' subdirectory
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    gt_dir = os.path.join(input_dir, 'ref')
    if not os.path.exists(gt_dir):
        sys.exit('Could not find GT file {0}'.format(gt_dir))

    filelist = [filename for filename in sorted(os.listdir(gt_dir)) if filename.endswith('txt')]

    # check complete and valid submission
    for filename in filelist:
        if not os.path.exists(os.path.join(submit_dir, filename)):
            sys.exit('Could not find submission file {0}'.format(filename))
        preds = np.loadtxt(os.path.join(submit_dir, filename), dtype=np.int)

        # x in [0, 1024), y in [0, 512)
        if not ((np.alltrue(np.logical_and(preds[:, 0] >= 0, preds[:, 0] < WIDTH))) and
                (np.alltrue(np.logical_and(preds[:, 1] >= 0, preds[:, 1] < HEIGHT)))):
            sys.exit('Invalid submission file {0}'.format(filename))

        # a pair of ceiling and floor junctions should share the same x coordinate
        if not np.alltrue(preds[::2, 0] == preds[1::2, 0]):
            sys.exit('Invalid submission file {0}'.format(filename))

        # x coordinates should be a monotonically non-decreasing sequence
        if not np.alltrue(preds[::2, 0][1:] - preds[::2, 0][:-1] >= 0):
            sys.exit('Invalid submission file {0}'.format(filename))

    # compute final results
    results = np.zeros((len(filelist), 3))
    for index, filename in enumerate(sorted(filelist)):
        preds = np.loadtxt(os.path.join(submit_dir, filename), dtype=np.int)
        gts = np.loadtxt(os.path.join(gt_dir, filename), dtype=np.int)
        Fs = evaluate(gts, preds, THRES)
        results[index] = np.array(
            (np.mean(Fs['junction']), np.mean(Fs['wireframe']), np.mean(Fs['plane'])))

    # write scores to a file named "scores.txt"
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        output_file.write("mean: %f\n" % np.mean(results))
        output_file.write("junction: %f\n" % np.mean(results[:, 0]))
        output_file.write("wireframe: %f\n" % np.mean(results[:, 1]))
        output_file.write("plane: %f\n" % np.mean(results[:, 2]))

    total_time = time.time() - time_start
    sys.stdout.write('Total time: %s\n' % (str(total_time), ))


if __name__ == '__main__':
    main()
