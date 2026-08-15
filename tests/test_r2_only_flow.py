"""Verify the R2-only activation flow using an in-memory S3 stub.

Covers the behaviour that `delete_local_after_sync=True` (the default) depends on:

1. tensors are deleted locally, index parquets + manifest are not
2. the store still opens from the local index
3. reading a missing chunk raises an actionable MissingChunkError naming `r2 pull`
4. a pull round-trips the activations bit-identically
5. a truncated/failed upload does NOT delete the local copy

Run from the repo root, after creating a small fixture run:

    python run.py extract_activations \
        --set model_name=hf-internal-testing/tiny-random-LlamaForCausalLM \
        --set emotions=joyful,sad,angry --set stories_per_emotion=100 \
        --set neutral_stories=100 --set run_name=deltest --set layer_spec=all \
        --set device_map=None --set dtype=float32 --set chunk_size=128
    PYTHONPATH=. python tests/test_r2_only_flow.py

Uses a stub rather than moto so it tests our logic, not an S3 emulator's IAM rules
(and so it is unaffected by real AWS_* vars in the shell).
"""
import pathlib, shutil, sys
import numpy as np
import core.r2 as r2mod
from core.r2 import R2Client, make_chunk_uploader
from core.activation_store import ActivationStore, MissingChunkError

class StubS3:
    """Minimal S3: put/get/head/list over a dict. Optionally truncates writes."""
    def __init__(self): self.objects = {}; self.truncate = False
    def upload_file(self, local, bucket, key):
        data = pathlib.Path(local).read_bytes()
        self.objects[key] = data[:7] if self.truncate else data
    def download_file(self, bucket, key, local):
        pathlib.Path(local).write_bytes(self.objects[key])
    def head_object(self, Bucket, Key):
        if Key not in self.objects: raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}
    def list_objects_v2(self, **kw):
        pre = kw.get("Prefix", "")
        items = [{"Key": k, "Size": len(v)} for k, v in sorted(self.objects.items())
                 if k.startswith(pre)]
        return {"Contents": items, "IsTruncated": False}

stub = StubS3()
client = R2Client(bucket="emotion-vector-perspectives",
                  endpoint_url="https://mock.r2.cloudflarestorage.com", client=stub)
R2Client.from_env = classmethod(lambda cls: client)

src = pathlib.Path("outputs/deltest/activations")
work = pathlib.Path("/tmp/r2flow/activations")
if work.parent.exists(): shutil.rmtree(work.parent)
work.parent.mkdir(parents=True); shutil.copytree(src, work)
prefix = "story-activations/deltest"

chunks = sorted(work.glob("shard*/chunk_*.safetensors"))
print(f"1. start: {len(chunks)} local tensor chunks")

upload = make_chunk_uploader(prefix, work, delete_local=True, verbose=False)
for t in chunks:
    upload([t, t.with_suffix(".index.parquet")])

left = list(work.glob("shard*/chunk_*.safetensors"))
idx  = list(work.glob("shard*/chunk_*.index.parquet"))
print(f"2. after verified upload: {len(left)} tensors local, {len(idx)} index local, "
      f"{len(stub.objects)} objects in R2")
assert not left, "tensors should be deleted locally"
assert len(idx) == len(chunks), "index parquets must remain for resume"
assert (work / "manifest.json").exists(), "manifest must remain local"

store = ActivationStore(work)
rows = store.subset(source="emotion", splits=["train"])
print(f"3. store still opens from local index: {len(store.index)} rows")

try:
    store.load_layer(1, rows); print("   !! FAIL expected MissingChunkError"); sys.exit(1)
except MissingChunkError as e:
    assert "r2 pull" in str(e)
    print(f"4. actionable error, first line: {str(e).splitlines()[0]}")

n = client.sync_down(work, prefix, verbose=False)
print(f"5. pulled back {n['downloaded']} objects")
got  = ActivationStore(work).load_layer(1, rows)
want = ActivationStore(src).load_layer(1, ActivationStore(src).subset(source="emotion", splits=["train"]))
assert got.shape == want.shape and np.array_equal(got, want), "round-trip mismatch!"
print(f"6. round-trip bit-identical: {got.shape}")

# Truncated upload must NOT delete the local copy.
stub.truncate = True
victim = sorted(work.glob("shard*/chunk_*.safetensors"))[0]
make_chunk_uploader(prefix, work, delete_local=True, verbose=False)(
    [victim, victim.with_suffix(".index.parquet")])
print(f"7. truncated-upload guard: local file kept = {victim.exists()}")
assert victim.exists(), "must NOT delete when remote size mismatches"

print("\nALL CHECKS PASSED")
