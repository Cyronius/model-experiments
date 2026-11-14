"""Restore the frozen projection plus a selected local correctness adapter."""
from pathlib import Path
import json
import torch
from broad_healing import OUT, RECIPE, SOURCE, sha
from local_projection_models import from_recipe
from local_projection_probe import load_source
from correctness_heal import install_lora, load_adapter


def load(checkpoint=None):
    # Match the construction and inference precision/thread convention.
    torch.set_num_threads(4)
    checkpoint=Path(checkpoint) if checkpoint else OUT/'selected'
    metadata=json.loads((checkpoint/'adapter.json').read_text())
    if metadata.get('recipe_sha256') != sha(RECIPE):
        raise ValueError('Projection recipe differs from trained adapter')
    provenance=json.loads((OUT/'reproducibility.json').read_text())
    for name, expected in provenance['source_hashes_verified'].items():
        if sha(Path(SOURCE)/name)!=expected:
            raise ValueError(f'Pinned source file differs: {name}')
    recipe=torch.load(RECIPE,map_location='cpu',weights_only=True)
    model=from_recipe(SOURCE,recipe,load_source)
    info=install_lora(model,metadata['rank'],metadata['alpha'])
    if info['modules']!=metadata['modules']:raise ValueError('LoRA module inventory differs')
    load_adapter(model,checkpoint)
    return model.eval()
