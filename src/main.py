import os

# Let pod env var control platform; default to cpu if unset
if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import time
import jax
import tensorstore as ts

# ── Configuration via env vars ──────────────────────────────────────
size = 2**15  # 32768
chunks_per_dim = int(os.environ.get("BENCH_CHUNKS_PER_DIM", "4"))
compressor = os.environ.get("BENCH_COMPRESSOR", "blosc")
iterations = int(os.environ.get("BENCH_ITERATIONS", "5"))
experiment = os.environ.get("BENCH_EXPERIMENT_NAME", "default")
data_copy_concurrency = int(os.environ.get("BENCH_DATA_COPY_CONCURRENCY", "0"))
s3_request_concurrency = int(os.environ.get("BENCH_S3_REQUEST_CONCURRENCY", "0"))

# ── Print config for reproducibility ────────────────────────────────
print("=" * 60, flush=True)
print(f"EXPERIMENT: {experiment}", flush=True)
print(f"  array_size:    {size}x{size}", flush=True)
print(f"  chunks_per_dim: {chunks_per_dim}  ({chunks_per_dim**2} total chunks)", flush=True)
chunk_bytes = (size // chunks_per_dim) ** 2 * 4
print(f"  chunk_size:    {chunk_bytes / 1024**2:.1f} MB", flush=True)
print(f"  compressor:    {compressor}", flush=True)
print(f"  iterations:    {iterations}", flush=True)
print(f"  data_copy_concurrency: {data_copy_concurrency or '(default)'}", flush=True)
print(f"  s3_request_concurrency: {s3_request_concurrency or '(default)'}", flush=True)
print(f"  TENSORSTORE_HTTP_THREADS: {os.environ.get('TENSORSTORE_HTTP_THREADS', '(default)')}", flush=True)
print(f"  TENSORSTORE_S3_REQUEST_CONCURRENCY: {os.environ.get('TENSORSTORE_S3_REQUEST_CONCURRENCY', '(default)')}", flush=True)
print(f"  JAX_PLATFORMS: {os.environ.get('JAX_PLATFORMS', '(auto)')}", flush=True)
print(f"  jax.devices: {jax.devices()}", flush=True)
print("=" * 60, flush=True)

# ── Build compressor metadata ───────────────────────────────────────
if compressor == "none":
    compressor_meta = None
elif compressor == "blosc":
    compressor_meta = {
        "id": "blosc",
        "cname": "lz4",
        "clevel": 5,
        "shuffle": 1,
    }
else:
    raise ValueError(f"Unknown compressor: {compressor}")


def make_spec(path, *, create=False):
    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "s3",
            "bucket": "ace-usw04a",
            "path": path,
            "endpoint": "http://ace-usw04a.cwlota.com",
            "host_header": "ace-usw04a.cwlota.com",
            "aws_region": "us-west-04a",
            "aws_credentials": {"type": "environment"},
        },
        "metadata": {
            "dtype": "<f4",
            "shape": [size, size],
            "chunks": [size // chunks_per_dim, size // chunks_per_dim],
            "compressor": compressor_meta,
        },
    }

    # Context overrides for concurrency tuning
    context = {}
    if data_copy_concurrency > 0:
        context["data_copy_concurrency"] = {"limit": data_copy_concurrency}
    if s3_request_concurrency > 0:
        context["s3_request_concurrency"] = {"limit": s3_request_concurrency}
    if context:
        spec["context"] = context

    if create:
        spec["create"] = True
        spec["open"] = True
    return spec


# ── Generate test data ──────────────────────────────────────────────
print("Generating test data...", flush=True)
t_gen = time.monotonic()
test_data = jax.random.normal(jax.random.PRNGKey(0), (size, size))
test_data.block_until_ready()
t_gen = time.monotonic() - t_gen
data_size_gb = test_data.itemsize * test_data.size / 1e9
print(f"Data size: {data_size_gb:.3f} GB  device: {test_data.devices()}  gen_time: {t_gen:.2f}s", flush=True)

# ── Run benchmark ───────────────────────────────────────────────────
throughputs = []
cumulative_start = time.monotonic()

for i in range(iterations):
    path = f"bench-{experiment}/iter-{i}/"
    t0 = time.monotonic()
    dataset = ts.open(make_spec(path, create=True)).result()
    dataset[...] = test_data
    elapsed = time.monotonic() - t0
    gbps = data_size_gb / elapsed
    throughputs.append(gbps)
    print(f"RESULT iter={i} elapsed={elapsed:.2f}s throughput={gbps:.3f} GB/s", flush=True)

cumulative_elapsed = time.monotonic() - cumulative_start
cumulative_gbps = (data_size_gb * iterations) / cumulative_elapsed

print("=" * 60, flush=True)
print(f"SUMMARY experiment={experiment}", flush=True)
print(f"  Best:       {max(throughputs):.3f} GB/s", flush=True)
print(f"  Worst:      {min(throughputs):.3f} GB/s", flush=True)
print(f"  Mean:       {sum(throughputs)/len(throughputs):.3f} GB/s", flush=True)
print(f"  Cumulative: {cumulative_gbps:.3f} GB/s ({cumulative_elapsed:.1f}s total)", flush=True)
print("=" * 60, flush=True)
