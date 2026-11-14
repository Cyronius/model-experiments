"""Disposable CUDA worker for one pod of the loop checks on the healed student (heal_probe.py with
heal_probe_loops_<experiment>_rules.json, the experiment named on the command line): heal_worker's
bundle and model checks, with the probe_loops_<experiment>/ folder in outputs/. No RunPod credential.

The evaluate stage checks the student and fits its fold, if it has one, which stays on the pod for
the generations; then each of the rules' generate arms in order."""
import json
from pathlib import Path
import sys

import heal_worker


def stages(experiment):
    rules = f'heal_probe_loops_{experiment}_rules.json'
    arms = json.loads((Path(__file__).resolve().parent / rules).read_text())['generate']['arms']
    probe = ['heal_probe.py']
    return [('evaluate', probe + ['evaluate', '--rules', rules], [])] + [
        (f'generate_{arm}', probe + ['generate', '--arm', arm, '--rules', rules], ['evaluate']) for arm in arms]


if __name__ == '__main__':
    heal_worker.run(stages(sys.argv[1]), f'probe_loops_{sys.argv[1]}')
