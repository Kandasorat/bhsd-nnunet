# CSAM and CSA-Net reproduction policy

## Rule

Attention experiments must be developed and reported in this order:

1. **Source-faithful reference**: reproduce the paper and official repository
   protocol as closely as the released material permits.
2. **BHSD port**: change only the dataset reader, class count, train/validation
   split, and CT intensity handling required to run the official protocol on
   BHSD. Record every unavoidable change.
3. **Harmonized nnU-Net adaptation**: use the common BHSD patch, optimizer/loss
   framework, maximum epochs, Early Stopping, and checkpoint policy for a
   controlled comparison with the nnU-Net baselines.
4. **Proposed improvement**: introduce new architectural or training changes
   only after the preceding references are available.

Results from different tiers must not share an experiment name or result
directory. An nnU-Net adaptation of an upstream architecture must not be called
an "official reproduction".

## CSAM upstream protocol

Primary sources:

- Paper: https://arxiv.org/abs/2311.04942
- Official repository: https://github.com/aL3x-O-o-Hung/CSAM
- Pinned source used here: `a0029206ef3b4147351813b7d67eb7b5964c8f33`

Defaults visible in the released `experiment.py` include:

- 150 epochs;
- sequence length 20;
- input size 128;
- U-Net mode;
- Adam with learning rate `1e-4`;
- CrossEntropyLoss;
- batch size 2;
- six U-Net levels, base width 64 in the released network initializer;
- validation after each epoch and model selection by validation Dice.

The released repository is not a complete turn-key training package:
`experiment.py` imports a data loader and network names that are not present
under those names in the repository, and the parser references a dataset
argument that is not declared. Consequently, any executable BHSD reproduction
must identify both the upstream defaults retained and the repairs/data adapters
added by this project.

## CSA-Net upstream protocol

Primary sources:

- Paper: https://arxiv.org/abs/2405.00130
- Official repository: https://github.com/mirthAI/CSA-Net
- Pinned source used here: `9be2dbe8d2247ab91d03f18bd8af92448a675ff9`

Defaults visible in the released `train.py`, `trainer.py`, and dataset code
include:

- previous, centre, and next slice as three separate inputs;
- R50-ViT-B/16 with the linked ImageNet-21k pretrained weights;
- input size 224;
- 40 epochs and batch size 16;
- SGD, learning rate `1e-3`, momentum 0.9, weight decay `1e-4`;
- loss `0.5 * CrossEntropy + 0.5 * Dice`;
- polynomial learning-rate decay with exponent 0.9;
- validation every five epochs after epoch 10 and at the final epoch;
- best checkpoint selected by validation Dice;
- seed 1234 and deterministic cuDNN settings.

The released CSA-Net attention code creates some cross-attention modules inside
`forward`. The vendored project source registers those modules in `__init__` and
removes a hard-coded CUDA placement. That correction is necessary for the
parameters to be trainable and checkpointed, but it must be disclosed as a
compatibility correction rather than silently described as byte-for-byte
official code.

## Status of the current configs

The following configs are **tier 3 nnU-Net adaptations**, not tier 1 official
protocol reproductions:

- `configs/csam_official_3slice.yaml`
- `configs/csam_official_3slice_binary.yaml`
- `configs/csam_official_volume32_fold0.yaml`
- `configs/csa_net_official_3slice_fold0.yaml`

Their filenames are retained temporarily for compatibility, but their
`experiment_name`, `protocol_tier`, and `source_faithful` fields are
authoritative. Do not submit the attention PBS pilots as official baselines
until separate source-faithful BHSD protocol configs and runners have passed a
fold-0 smoke test.
