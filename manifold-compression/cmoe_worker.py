"""Disposable CUDA worker for the CMoE router comparison (cmoe_router.py evaluate): heal_worker's
bundle and model checks, with the cmoe/ folder in outputs/. No RunPod credential."""
import heal_worker

STAGES = [('evaluate', ['cmoe_router.py', 'evaluate'], [])]

if __name__ == '__main__':
    heal_worker.run(STAGES, 'cmoe')
