# Context, boundaries, and what the benchmark measures

## Why a halo is necessary

A 3 × 3 stride-one convolution sees one pixel beyond its output location. Two such
layers have receptive-field radius two. A core tile therefore needs two real pixels
of context on each internal edge. The model runs on the expanded tile; its halo
predictions are cropped before the core is accumulated.

At a true image edge, the input crop is clipped to the image. The model's own layer
padding reproduces the whole-image boundary. Padding the raw image with zeros out to
every halo would be a different computation for stacked layers: the intermediate
activations outside the real image can affect the next layer.

Overlap and halo solve different problems. Halo supplies receptive-field context.
Overlap averages competing predictions. Overlap alone cannot restore context that the
model never saw. The seam diagnostic holds the model and image fixed and changes only
halo size.

## What the API guarantees

`tiled_predict` accepts a finite CPU float32 CHW image and an eval-mode PyTorch module
that returns finite floating-point BCHW logits at the same spatial resolution. It
returns float32 CHW logits. Every output pixel has a strictly positive accumulated
weight. Final argmax should happen after blending logits, not before.

Tiles are generated in raster order. Only consecutive equal-shaped tiles are batched;
there is no growing collection of partial shape buckets. Edge inputs can have different
shapes and cause early batch flushes. Peak activations depend on expanded tile size,
batch size and the model. The CPU source image, full output accumulator, and weight
denominator remain resident. This is **not out-of-core raster I/O**.

The materialized control deliberately copies all expanded input tiles before running
the same inference and stitching logic. It isolates one common avoidable allocation.
Whole-frame inference has a different activation allocation pattern and does no tile
stitching. A smaller RSS is not automatically lower latency.

## Parity is conditional

Whole-image equality is expected only for spatially preserving, translation-equivariant
local operations with sufficient context and compatible padding. The included two-layer
CNN has radius two. Floating-point differences are checked with rtol=1e-5 and atol=1e-6.

Global attention, global pooling, spatial instance/group normalization, training-mode
batch normalization, encoder stride alignment and interpolation can change results.
The test suite includes a global-mean operation that intentionally does **not** match
whole-frame execution. The library cannot infer a correct halo for an arbitrary model.
Users must validate their own model against a tractable whole-frame reference.

CPU float32 is the validated implementation. CUDA, mixed precision, distributed
execution, georeferenced output and multi-scale/strided model adapters are not supported
by this release. No GPU speedup is claimed.

## Benchmark protocol

Each mode starts a separate Python process with identical model seed, image seed,
dimensions, weights and thread limit. Input and model setup are outside the timed
region. The first call is recorded separately as cold; repeated subsequent calls
produce raw warm times and a median. Timings include validation, tile planning,
stacking, model execution and stitching for tiled modes. The whole-frame control
times its direct forward pass. Import/startup, disk output and numerical comparison
are excluded. No CUDA synchronization is relevant to these CPU measurements.

A monitor thread samples process RSS every 5 ms during all timed calls. The report
contains setup RSS, sampled maximum and the increase above setup. Sampling can miss
short peaks; these figures are estimates, not allocator high-water marks. CPU time
covers the worker process including its monitoring thread. Workers do not spawn child
workers, and each command has a 300-second timeout. The parent compares saved logits
after each worker exits and fails the benchmark if tolerance is exceeded.

The fixture is random synthetic RGB input and an **untrained** convolutional model.
This measures inference mechanics and numerical equivalence, not semantic segmentation
accuracy. There are no real-world model-quality claims.

References: [PyTorch benchmarking guide](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
and [inference-mode semantics](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html).
Inference mode disables autograd; calling `model.eval()` is still necessary.
