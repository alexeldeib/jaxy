# TensorStore S3 Write Throughput Optimization

Benchmarking and tuning guide for writing large JAX arrays to S3-backed Zarr via TensorStore.

## Recommended Configuration

For writing 32768×32768 float32 arrays (~4.3 GB each), the following configuration achieves **5–6 GB/s** sustained write throughput:

| Parameter | Value | How to Set |
|---|---|---|
| Chunk size | 4 MB (32×32 chunks per dim) | `metadata.chunks` in TensorStore spec |
| Compressor | `None` | `metadata.compressor: null` in TensorStore spec |
| HTTP threads | 64 | `TENSORSTORE_HTTP_THREADS=64` env var |
| S3 request concurrency | 512 | `TENSORSTORE_S3_REQUEST_CONCURRENCY=512` env var |
| CPU cores | 16 | Pod resource requests |

### Minimal Python Example

```python
import os
import jax
import tensorstore as ts

size = 2**15  # 32768
chunks_per_dim = 32

spec = {
    "driver": "zarr",
    "kvstore": {
        "driver": "s3",
        "bucket": "ace-usw04a",
        "path": "my-array/",
        "endpoint": "http://ace-usw04a.cwlota.com",
        "host_header": "ace-usw04a.cwlota.com",
        "aws_region": "us-west-04a",
        "aws_credentials": {"type": "environment"},
    },
    "metadata": {
        "dtype": "<f4",
        "shape": [size, size],
        "chunks": [size // chunks_per_dim, size // chunks_per_dim],
        "compressor": None,
    },
    "create": True,
    "open": True,
}

data = jax.random.normal(jax.random.PRNGKey(0), (size, size))
dataset = ts.open(spec).result()
dataset[...] = data
```

### Required Environment Variables

```bash
export TENSORSTORE_HTTP_THREADS=64
export TENSORSTORE_S3_REQUEST_CONCURRENCY=512
```

## Benchmark Results

All experiments write a 32768×32768 float32 array (~4.3 GB) to S3-backed Zarr over 5 iterations. "Steady-state" excludes the first iteration (which pays connection establishment overhead).

### Baseline

The original script used default TensorStore settings with 4 CPU cores, 4×4 chunking (256 MB chunks), and default Blosc compression. No TensorStore environment variables were set.

| | Value |
|---|---|
| Reported throughput | **~0.9 GB/s** |
| CPU | 4 |
| Chunks | 4×4 = 16 (256 MB each) |
| HTTP threads | 4 (default) |
| S3 concurrency | 32 (default) |
| Compressor | blosc (default) |

### CPU Experiments

| Experiment | CPU | Chunks | Chunk Size | HTTP Threads | S3 Concurrency | Compressor | Steady-state GB/s |
|---|---|---|---|---|---|---|---|
| baseline | 4 | 4×4 = 16 | 256 MB | 4 | 32 | blosc | ~1.2 |
| nocomp | 4 | 4×4 = 16 | 256 MB | 4 | 32 | none | ~1.15 |
| threads16 | 4 | 4×4 = 16 | 256 MB | 16 | 64 | none | ~1.1 |
| chunks64-cpu8 | 8 | 8×8 = 64 | 64 MB | 32 | 128 | none | ~2.5 |
| chunks256-cpu16 | 16 | 16×16 = 256 | 16 MB | 32 | 256 | none | ~4–5 |
| max-parallel | 16 | 32×32 = 1024 | 4 MB | 64 | 512 | none | ~5 |

### GPU Experiments

| Experiment | GPU | CPU | Chunks | Chunk Size | HTTP Threads | S3 Concurrency | Steady-state GB/s |
|---|---|---|---|---|---|---|---|
| gpu-chunks256 | 1 | 16 | 16×16 = 256 | 16 MB | 32 | 256 | ~5 |
| gpu-max-parallel | 1 | 16 | 32×32 = 1024 | 4 MB | 64 | 512 | **~5.7** |

## Key Findings

### What matters

1. **Chunk count + CPU cores** are the dominant factors. More chunks create more concurrent S3 PUT operations; more CPU cores feed those operations faster via TensorStore's `data_copy_concurrency` (which defaults to CPU count).

2. **HTTP thread count** must match chunk parallelism. `TENSORSTORE_HTTP_THREADS` caps how many HTTP requests can execute simultaneously. Set it to at least the number of chunks, or higher.

3. **S3 request concurrency** should exceed chunk count. `TENSORSTORE_S3_REQUEST_CONCURRENCY` controls TensorStore's S3 upload queue depth. Set it to 2–4× the chunk count.

### What doesn't matter

1. **Compression** has no effect on random float32 data. Blosc detects incompressible data quickly and short-circuits, so the overhead is negligible. For compressible data (e.g., sparse arrays), compression may still be worthwhile to reduce transfer volume.

2. **HTTP threads alone** (without more chunks/CPU) don't help. Adding threads to a 4-CPU / 16-chunk workload had no effect — the bottleneck was upstream in chunk encoding, not in the HTTP client.

3. **GPU vs CPU source arrays** — GPU-resident arrays perform as well or slightly better than CPU arrays. The device-to-host PCIe transfer (~25 GB/s) is not a bottleneck relative to S3 upload throughput.

## Bottleneck Model

```
JAX Array → [data_copy_concurrency: chunk encoding] → [HTTP threads: upload queue] → S3
              ↑ scales with CPU cores                   ↑ TENSORSTORE_HTTP_THREADS
              ↑ more chunks = more parallelism          ↑ TENSORSTORE_S3_REQUEST_CONCURRENCY
```

The baseline configuration (4 CPU, 16 chunks, 4 HTTP threads) created a bottleneck at every stage. The optimized configuration removes all three:

- 1024 chunks → encoding and upload are highly parallel
- 16 CPU cores → `data_copy_concurrency` can saturate the pipeline
- 64 HTTP threads / 512 S3 concurrency → no queuing in the HTTP layer

## Reproducing

The benchmark script (`src/main.py`) is parameterized via environment variables. Pod manifests for each experiment are in `k8s/experiments/`.

```bash
# Update the ConfigMap with the benchmark script
kubectl create configmap jaxy-script --from-file=main.py=src/main.py -n <namespace>

# Run all experiments
kubectl apply -f k8s/experiments/

# Collect results
kubectl logs <pod-name> -n <namespace> | grep -E "RESULT|SUMMARY|Best|Worst|Mean|Cumulative"
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BENCH_CHUNKS_PER_DIM` | `4` | Chunks per array dimension (total chunks = value²) |
| `BENCH_COMPRESSOR` | `blosc` | `blosc` or `none` |
| `BENCH_ITERATIONS` | `5` | Number of write iterations |
| `BENCH_EXPERIMENT_NAME` | `default` | Experiment name (used in S3 path prefix and logs) |
| `TENSORSTORE_HTTP_THREADS` | `4` (TensorStore default) | HTTP client thread pool size |
| `TENSORSTORE_S3_REQUEST_CONCURRENCY` | `32` (TensorStore default) | Max concurrent S3 requests |
