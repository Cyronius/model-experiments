"""Exercise real trainer interruption/resume with tiny CPU models and synthetic data."""
import json
from pathlib import Path
import random
import runpy
import signal
import sys
import tempfile
import types

import torch

SCRIPT = Path(__file__).with_name("distill_whole.py")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16)

    def save_pretrained(self, directory, state_dict):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, directory / "weights.pt")


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        return types.SimpleNamespace(input_ids=list(range(16)) * 16)

    def save_pretrained(self, directory):
        pass


def run(output, *, resume=None, interrupt=False, start_tokens=0):
    torch.manual_seed(123)
    random.seed(123)

    def load_model(name, **kwargs):
        model = TinyModel()
        if name == "student":
            # Match the actual common.load_model result shape.
            embedding = model.model
            class Body(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = embedding

                def forward(self, input_ids):
                    return types.SimpleNamespace(last_hidden_state=self.embed(input_ids))
            model.model = Body()
        else:
            embedding = model.model
            class TeacherBody(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = embedding

                def forward(self, input_ids):
                    return types.SimpleNamespace(last_hidden_state=self.embed(input_ids))
            model.model = TeacherBody()
        return model, TinyTokenizer()

    common = types.ModuleType("common")
    common.load_model = load_model
    common.set_threads = lambda n: torch.set_num_threads(n)
    common._wikitext = lambda split: "synthetic"
    common._ultrachat_text = lambda: "unused"
    common.build_data = lambda *a, **k: {"fit": [("prose", list(range(8)))] * 32,
                                        "eval": [("prose", list(range(8)))] * 2}
    common.save_json = lambda obj, path: Path(path).write_text(json.dumps(obj))
    sys.modules["common"] = common
    sys.argv = [str(SCRIPT), "--teacher", "teacher", "--student", "student", "--out", str(output),
                "--tokens", "128", "--T", "8", "--rows-per-step", "2", "--micro", "1",
                "--chunk", "4", "--corpus", "tiny", "--eval-every", "32", "--save-every", "32",
                "--checkpoints", "64,128", "--threads", "1"]
    if resume:
        sys.argv += ["--resume-state", str(resume)]
    if start_tokens:
        sys.argv += ["--start-tokens", str(start_tokens)]
    original_step = torch.optim.AdamW.step
    calls = 0

    def step(optimizer, *a, **k):
        nonlocal calls
        result = original_step(optimizer, *a, **k)
        calls += 1
        if interrupt and calls == 3:
            signal.raise_signal(signal.SIGTERM)
        return result

    torch.optim.AdamW.step = step
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as error:
        assert interrupt and error.code == 75, error
    finally:
        torch.optim.AdamW.step = original_step


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    run(root / "full")
    run(root / "split", interrupt=True)
    interrupted = torch.load(root / "split/resume_latest.pt", weights_only=False)
    assert interrupted["tokens"] == 48 and interrupted["cursor"] == 6
    run(root / "split", resume=root / "split/resume_latest.pt")
    full = torch.load(root / "full/resume_latest.pt", weights_only=False)
    split = torch.load(root / "split/resume_latest.pt", weights_only=False)
    assert full["tokens"] == split["tokens"] == 128
    assert full["cursor"] == split["cursor"] == 16
    for key in full["student"]:
        torch.testing.assert_close(full["student"][key], split["student"][key], rtol=0, atol=0)
    assert full["history"]["train"] == split["history"]["train"]
    assert all(int(s["step"]) == 8 for s in split["optimizer"]["state"].values())
    run(root / "weights_only", start_tokens=48)
    weights = torch.load(root / "weights_only/resume_latest.pt", weights_only=False)
    assert weights["history"]["train"][0]["tokens"] == 64
    assert weights["history"]["train"][0]["lr"] == full["history"]["train"][3]["lr"]
    assert weights["tokens"] == 128 and weights["history"]["resumes"][0]["optimizer_restored"] is False
    print("PASS: interrupted full-state resume matches uninterrupted weights and losses exactly; "
          "weights-only continuation preserves token count, corpus offset, and LR schedule")
