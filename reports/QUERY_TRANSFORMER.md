# Query transformer

The optional model in `training/query_transformer.py` implements highlighted row scoring and sequential double removal on temporary boards. It has **79,148 trainable parameters** at width 64, two attention layers, four heads, and feed-forward width 128. The previous flat transformer has 6,838,443 parameters.

The shared scalar take head scores both row proposals and individual double decisions. Row scores use a softmax; double decisions use conditional Bernoulli probabilities in the engine line order. Mandatory singles and earlier chosen doubles disappear from subsequent query boards, with own stones returned to reserve. Selected-row and current-double highlights remain explicit. The engine still receives complete legal action probabilities; there is no learned 2,730-output projection.

The encoder mean-pools its 37 tokens before 42 insertion outputs, the shared take output, and the value head. Therefore the parameter reduction includes pooling instead of flattening, not just factorizing captures. Nine ordinary feature planes remain the public input; three query channels are constructed internally.

CPU tests cover normalized legal policies, gradients, checkpoint round trips, conditional depth, and temporary-board outcomes against the authoritative engine. A separate review matched reconstructed legal actions over 40,000 random-play decisions. This is an untrained implementation, not playing-strength evidence.

The optional `--kind transformer --head query --width 64 --blocks 2` training integration is queued after the existing frozen architecture comparison. Its staged patch is `patches/enable_query_transformer.patch`. It does not replace the transformer currently under comparison.

This first implementation constructs queries on CPU and batches row/prefix queries within each position. Additional inference passes and host transfers may be expensive; throughput has not been benchmarked.

## Larger preset for parameter comparisons

`ops/query_transformer_matched_config.json` selects width 256, two layers, four
attention heads and feed-forward width 512. It contains **1,102,124** trainable
parameters, between the existing MLP (979,371) and CNN (1,163,523). Feed-forward
width now scales as twice token width; the original width-64 query model remains
unchanged at 79,148 parameters. This preset is implemented but untrained and is
not substituted into the in-progress comparison. Its forward pass, legal-policy
normalization and finite backward gradients were checked on CPU.
