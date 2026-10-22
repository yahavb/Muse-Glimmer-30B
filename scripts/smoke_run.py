"""CPU smoke test of run_muse_glimmer.py end to end, no Trainium needed.

Saves a tiny random-weight Muse-Glimmer (real processor/tokenizer) and runs the real
script with stand-ins: torch_neuronx stub, 'neuron' compile backend = aot_eager,
gloo process group. Checks flow/shapes/recompiles, not answer quality.

    torchrun --nproc_per_node 1 scripts/smoke_run.py --config cfg.json [--script profile_vision.py] -- <args>
"""
import json, os, runpy, sys, types

import torch

argv = sys.argv[1:]
cfg_path = argv[argv.index("--config") + 1]
run_args = argv[argv.index("--") + 1:]
script = argv[argv.index("--script") + 1] if "--script" in argv[:argv.index("--")] else "run_muse_glimmer.py"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# stand-ins for the Neuron pieces
sys.modules["torch_neuronx"] = types.ModuleType("torch_neuronx")
torch.neuron = types.SimpleNamespace(synchronize=lambda: None)
torch._C._get_accelerator = lambda: torch.device("cpu")
torch.accelerator.set_device_index = lambda i: None
torch._dynamo.register_backend(name="neuron")(torch._dynamo.lookup_backend("aot_eager"))
import transformers.distributed.utils as du
du.get_distributed_backend = lambda device_type: "gloo"

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
path = "/tmp/muse_tiny_smoke"
if not os.path.exists(os.path.join(path, "config.json")):
    cfg = json.load(open(cfg_path)); t = cfg["text_config"]
    t.update(num_hidden_layers=4, hidden_size=256, intermediate_size=512, num_attention_heads=4,
             num_key_value_heads=2, head_dim=64, layer_types=t["layer_types"][:4],
             layer_rope_theta=t["layer_rope_theta"][:4], sliding_window=16)
    v = cfg["vision_config"]
    v.update(num_hidden_layers=4, hidden_size=128, intermediate_size=256, num_attention_heads=4,
             layer_types=v["layer_types"][:4])
    cfg.update(out_hidden_size=128 * v["merge_size"] ** 2, projector_hidden_size=256)
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(AutoConfig.for_model(**cfg)).save_pretrained(path)
    AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B").save_pretrained(path)
os.environ["MUSE_MODEL_ID"] = path
os.environ.setdefault("MUSE_DEVICE", "cpu")

sys.argv = [os.path.join(root, script)] + run_args
sys.path.insert(0, root)
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as e:
    if e.code not in (0, None):
        raise
