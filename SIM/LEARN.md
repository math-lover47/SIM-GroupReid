**Current Architecture**

This repo is a FastReID fork with SIM-specific changes injected at four layers.

1. Entrypoint and orchestration. Training starts in [tools/train_net.py](/home/arsen/group_reid/methods/SIM/tools/train_net.py#L16), which builds config, sets seed, and hands control to `DefaultTrainer`. Most of the training loop is inherited FastReID boilerplate rather than SIM-specific logic.

2. Data contract. The important custom step is [fastreid/data/common.py](/home/arsen/group_reid/methods/SIM/fastreid/data/common.py#L83). Each sample is not just one image. It produces:
   - full group image: `images`
   - per-person crops: `images_p`
   - per-person layout coordinates: `layout`
   - group label: `targets`
   - person labels: `targets_p`
   - optional social interaction weights: `avg_probs`

3. Dataset preprocessing. SIM depends on an offline preprocessing stage in [fastreid/data/transforms/CSG_interaction.py](/home/arsen/group_reid/methods/SIM/fastreid/data/transforms/CSG_interaction.py#L174), which computes pose-based interaction matrices and stores enhanced pickle annotations. Dataset loaders like [fastreid/data/datasets/CSG.py](/home/arsen/group_reid/methods/SIM/fastreid/data/datasets/CSG.py#L41) then read those enhanced labels and reduce each interaction matrix to per-member `avg_probs`.

4. Model architecture. The core SIM model is the `Baseline` meta-arch in [fastreid/modeling/meta_arch/baseline.py](/home/arsen/group_reid/methods/SIM/fastreid/modeling/meta_arch/baseline.py#L19), which uses:
   - one shared backbone
   - one group head `heads_g`
   - one person head `heads_p`

The configured backbone is `build_gvit_backbone` from [fastreid/modeling/backbones/group_vit.py](/home/arsen/group_reid/methods/SIM/fastreid/modeling/backbones/group_vit.py#L692). That backbone has two stages:

- `p_ViT`: extracts per-person appearance tokens
- `g_ViT`: aggregates those person tokens into a group token using layout encoding, member uncertainty, and attention

The SIM-specific mechanisms are concentrated here:

- social-prior attention support in `Attention.forward(..., avg_probs=...)` at [fastreid/modeling/backbones/group_vit.py](/home/arsen/group_reid/methods/SIM/fastreid/modeling/backbones/group_vit.py#L69)
- layout uncertainty modeling at [fastreid/modeling/backbones/group_vit.py](/home/arsen/group_reid/methods/SIM/fastreid/modeling/backbones/group_vit.py#L500)
- person-to-group aggregation in `GVit.forward` at [fastreid/modeling/backbones/group_vit.py](/home/arsen/group_reid/methods/SIM/fastreid/modeling/backbones/group_vit.py#L633)

Two repo realities to keep in mind while reviewing:

- Paths are still hardcoded for a Windows environment in [fastreid/data/datasets/CSG.py](/home/arsen/group_reid/methods/SIM/fastreid/data/datasets/CSG.py#L22) and [configs/CSG/bagtricks_gvit.yml](/home/arsen/group_reid/methods/SIM/configs/CSG/bagtricks_gvit.yml#L6).
- There is an architectural mismatch: `CommDataset` emits `avg_probs`, and `group_vit` can use them, but `Baseline.forward` never passes `avg_probs` into the backbone in [fastreid/modeling/meta_arch/baseline.py](/home/arsen/group_reid/methods/SIM/fastreid/modeling/meta_arch/baseline.py#L114). So part of the intended SIM attention path may not actually be active.

**How To Review The Repo**

Use this order. It minimizes confusion and separates framework code from the SIM additions.

1. Start with the user story.
   Read [README.md](/home/arsen/group_reid/methods/SIM/README.md) and one active config like [configs/CSG/bagtricks_gvit.yml](/home/arsen/group_reid/methods/SIM/configs/CSG/bagtricks_gvit.yml#L1).
   Goal: know what dataset, backbone, losses, and output directory define the “real” experiment.

2. Trace one training batch end to end.
   Read:
   - [tools/train_net.py](/home/arsen/group_reid/methods/SIM/tools/train_net.py#L29)
   - [fastreid/engine/defaults.py](/home/arsen/group_reid/methods/SIM/fastreid/engine/defaults.py)
   - [fastreid/engine/train_loop.py](/home/arsen/group_reid/methods/SIM/fastreid/engine/train_loop.py#L174)
     Goal: understand exactly when config, dataloader, model, optimizer, hooks, eval, and checkpoints are built.

3. Understand the sample format before the model.
   Read:
   - [fastreid/data/build.py](/home/arsen/group_reid/methods/SIM/fastreid/data/build.py#L22)
   - [fastreid/data/common.py](/home/arsen/group_reid/methods/SIM/fastreid/data/common.py#L83)
   - one dataset loader first: [fastreid/data/datasets/CSG.py](/home/arsen/group_reid/methods/SIM/fastreid/data/datasets/CSG.py#L16)
     Goal: be able to describe every tensor in one batch and where it came from.

4. Understand the offline preprocessing.
   Read [fastreid/data/transforms/CSG_interaction.py](/home/arsen/group_reid/methods/SIM/fastreid/data/transforms/CSG_interaction.py#L1).
   Goal: know what the “social interaction” signal really is, how pose is used, and how the enhanced pickle files are produced.

5. Read the model at the meta-architecture level.
   Read [fastreid/modeling/meta_arch/baseline.py](/home/arsen/group_reid/methods/SIM/fastreid/modeling/meta_arch/baseline.py#L19).
   Goal: understand the split between group supervision and person supervision.

6. Read `group_vit.py` in this order.
   - `Attention` and `Block`
   - `p_ViT`
   - `g_ViT`
   - `GVit.forward`
   - `build_gvit_backbone`
     Goal: map each paper concept to code:
   - SICM: preprocessing / interaction matrix generation
   - SPAM-like weighting: attention with `avg_probs`
   - SLVM/SGL-like layout modeling: `layout_uncertainty_modeling`
   - final group representation: `g_ViT` cls token

7. Read the heads and losses.
   Read:
   - [fastreid/modeling/heads/embedding_head.py](/home/arsen/group_reid/methods/SIM/fastreid/modeling/heads/embedding_head.py#L11)
   - loss files under [fastreid/modeling/losses](/home/arsen/group_reid/methods/SIM/fastreid/modeling/losses)
     Goal: understand what is optimized for group tokens vs person tokens.

8. Read evaluation last.
   Read [fastreid/evaluation/reid_evaluation.py](/home/arsen/group_reid/methods/SIM/fastreid/evaluation/reid_evaluation.py#L18).
   Goal: know what metrics are actually reported and whether CSG has custom rank handling.

**Concrete Review Checklist**

Use these questions as you go:

- What parts are inherited FastReID, and what parts are SIM-specific?
- What exact tensors represent “social interaction”?
- Is every paper concept actually connected in code?
- Are enhanced annotations required for both train and test?
- Are group labels and person labels both used everywhere intended?
- Are there any dead paths, placeholders, or partially integrated features?
- Which files are repo-critical, and which are generic framework support?

If you want, I can turn this into a tighter deliverable next: either a one-page architecture map, or a file-by-file reading guide with estimated time per section.
