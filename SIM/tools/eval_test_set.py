import argparse
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw

from methods.SIM.fastreid.config import get_cfg
from methods.SIM.fastreid.data import build_reid_test_loader
from methods.SIM.fastreid.evaluation.rank import evaluate_rank
from methods.SIM.fastreid.modeling.meta_arch import build_model
from methods.SIM.fastreid.utils.checkpoint import Checkpointer
from methods.SIM.fastreid.utils.compute_dist import build_dist
from methods.SIM.fastreid.utils.file_io import PathManager
from methods.SIM.fastreid.utils.logger import setup_logger


def default_argument_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate SIM on the test set and save retrieval visualizations"
    )
    parser.add_argument("--config-file", required=True, help="Path to model config")
    parser.add_argument(
        "--path-config",
        default="configs/local_paths.yml",
        help="Optional path override config merged after --config-file",
    )
    parser.add_argument("--weights", default="", help="Checkpoint to evaluate")
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Override DATASETS.TESTS and DATASETS.NAMES with a single dataset name",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--num-vis", type=int, default=20, help="Number of queries to visualize")
    parser.add_argument("--vis-topk", type=int, default=3, help="Top-k gallery results to draw")
    parser.add_argument(
        "--vis-rank-sort",
        choices=["ascending", "descending", "random"],
        default="descending",
        help="How to select visualization queries using AP",
    )
    parser.add_argument(
        "--metric",
        default="",
        help="Override TEST.METRIC for distance computation",
    )
    parser.add_argument("--flip-test", action="store_true", help="Enable flipped-image inference")
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Additional config overrides KEY VALUE",
    )
    return parser


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.path_config and os.path.exists(args.path_config):
        cfg.merge_from_file(args.path_config)
    if args.dataset_name:
        cfg.DATASETS.NAMES = (args.dataset_name,)
        cfg.DATASETS.TESTS = (args.dataset_name,)
    if args.metric:
        cfg.TEST.METRIC = args.metric
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def run_inference(model, data_loader, flip_test=False):
    model.eval()
    features = []
    pids = []
    camids = []
    img_paths = []

    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            if flip_test:
                flipped_inputs = dict(inputs)
                flipped_inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(flipped_inputs)
                outputs = (outputs + flip_outputs) / 2

            features.append(outputs.cpu())
            pids.append(inputs["targets"].cpu())
            camids.append(inputs["camids"].cpu())
            img_paths.extend(list(inputs["img_paths"]))

    return {
        "features": torch.cat(features, dim=0),
        "pids": torch.cat(pids, dim=0).numpy(),
        "camids": torch.cat(camids, dim=0).numpy(),
        "img_paths": img_paths,
    }


def compute_metrics(cfg, features, pids, camids, num_query):
    query_features = features[:num_query]
    gallery_features = features[num_query:]
    query_pids = pids[:num_query]
    gallery_pids = pids[num_query:]
    query_camids = camids[:num_query]
    gallery_camids = camids[num_query:]

    dist = build_dist(query_features, gallery_features, cfg.TEST.METRIC)
    use_csg = any("CSG" in name for name in cfg.DATASETS.TESTS)
    cmc, all_ap, all_inp = evaluate_rank(
        dist,
        query_pids,
        gallery_pids,
        query_camids,
        gallery_camids,
        use_csg=use_csg,
        use_cython=False,
    )

    metrics = OrderedDict()
    for rank in [1, 5, 10]:
        if rank - 1 < len(cmc):
            metrics[f"Rank-{rank}"] = float(cmc[rank - 1] * 100.0)
    metrics["mAP"] = float(np.mean(all_ap) * 100.0)
    metrics["mINP"] = float(np.mean(all_inp) * 100.0)
    metrics["metric"] = float((np.mean(all_ap) + cmc[0]) / 2.0 * 100.0)

    return {
        "metrics": metrics,
        "dist": dist,
        "all_ap": np.asarray(all_ap),
        "query_features": query_features,
        "gallery_features": gallery_features,
        "query_pids": query_pids,
        "gallery_pids": gallery_pids,
        "query_camids": query_camids,
        "gallery_camids": gallery_camids,
    }


def compute_rank_lists(dist, q_pids, g_pids, q_camids, g_camids, use_csg=False):
    indices = np.argsort(dist, axis=1)
    if use_csg:
        indices = indices[:, 1:]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    rank_lists = []
    aps = []
    for q_idx in range(len(q_pids)):
        order = indices[q_idx]
        remove = (g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx])
        keep = np.invert(remove)
        filtered_order = order[keep]
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)]) * raw_cmc
        ap = float(tmp_cmc.sum() / num_rel)
        rank_lists.append((q_idx, filtered_order, raw_cmc))
        aps.append(ap)
    return rank_lists, np.asarray(aps)


def select_queries(rank_lists, aps, mode, num_vis):
    if not rank_lists:
        return []
    order = np.arange(len(rank_lists))
    if mode == "ascending":
        order = np.argsort(aps)
    elif mode == "descending":
        order = np.argsort(-aps)
    else:
        order = list(order)
        random.shuffle(order)
    return [(rank_lists[i], float(aps[i])) for i in order[:num_vis]]


def load_panel_image(path, size):
    image = Image.open(path).convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize(size, resampling.BICUBIC)
    return image


def draw_text(draw, xy, text, fill):
    draw.text(xy, text, fill=fill)


def build_retrieval_strip(query_path, gallery_paths, gallery_hits, gallery_scores):
    panel_w, panel_h = 160, 320
    margin = 16
    label_h = 48
    total = 1 + len(gallery_paths)
    canvas = Image.new(
        "RGB",
        (margin + total * (panel_w + margin), panel_h + label_h + 2 * margin),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    items = [(query_path, "query", None)] + list(zip(gallery_paths, gallery_scores, gallery_hits))
    for idx, item in enumerate(items):
        x = margin + idx * (panel_w + margin)
        y = margin
        image_path, tag, hit = item
        panel = load_panel_image(image_path, (panel_w, panel_h))
        canvas.paste(panel, (x, y))
        border_color = (0, 0, 0) if hit is None else ((0, 170, 0) if hit else (220, 0, 0))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=border_color, width=4)
        if hit is None:
            text = "Query"
        else:
            text = f"d={tag:.3f} {'T' if hit else 'F'}"
        draw_text(draw, (x, y + panel_h + 8), text, fill=(0, 0, 0))

    return canvas


def save_retrieval_strip(output_path, query_path, gallery_paths, gallery_hits, gallery_scores):
    canvas = build_retrieval_strip(query_path, gallery_paths, gallery_hits, gallery_scores)
    canvas.save(output_path)


def save_retrieval_grid(output_path, rows):
    if not rows:
        return

    row_images = [
        build_retrieval_strip(
            row["query_path"],
            row["gallery_paths"],
            row["gallery_hits"],
            row["gallery_scores"],
        )
        for row in rows
    ]
    max_width = max(image.width for image in row_images)
    total_height = sum(image.height for image in row_images) + 16 * (len(row_images) + 1)
    canvas = Image.new("RGB", (max_width + 32, total_height), color=(255, 255, 255))

    y = 16
    for index, image in enumerate(row_images):
        canvas.paste(image, (16, y))
        y += image.height + 16

    canvas.save(output_path)


def save_visualizations(
    output_dir,
    rank_lists,
    aps,
    query_paths,
    gallery_paths,
    dist,
    num_vis,
    vis_topk,
    vis_rank_sort,
):
    vis_dir = os.path.join(output_dir, "retrieval_topk")
    PathManager.mkdirs(vis_dir)
    selected = select_queries(rank_lists, aps, vis_rank_sort, num_vis)
    manifest = []
    grid_rows = []

    for vis_idx, ((q_idx, filtered_order, raw_cmc), ap_value) in enumerate(selected):
        top_order = filtered_order[:vis_topk]
        top_hits = [bool(x) for x in raw_cmc[:vis_topk]]
        top_scores = [float(dist[q_idx, g_idx]) for g_idx in top_order]
        out_path = os.path.join(vis_dir, f"{vis_idx:03d}_q{q_idx}.jpg")
        save_retrieval_strip(
            out_path,
            query_paths[q_idx],
            [gallery_paths[g_idx] for g_idx in top_order],
            top_hits,
            top_scores,
        )
        manifest.append(
            {
                "query_index": int(q_idx),
                "query_path": query_paths[q_idx],
                "ap": ap_value,
                "output": out_path,
                "gallery_paths": [gallery_paths[g_idx] for g_idx in top_order],
                "gallery_hits": top_hits,
                "gallery_scores": top_scores,
            }
        )
        grid_rows.append(
            {
                "query_path": query_paths[q_idx],
                "gallery_paths": [gallery_paths[g_idx] for g_idx in top_order],
                "gallery_hits": top_hits,
                "gallery_scores": top_scores,
            }
        )

    with open(os.path.join(output_dir, "retrieval_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    save_retrieval_grid(os.path.join(output_dir, "retrieval_grid.jpg"), grid_rows)


def main(args):
    PathManager.mkdirs(args.output_dir)
    logger = setup_logger(args.output_dir, name="fastreid")
    cfg = setup_cfg(args)
    logger.info("Running evaluation with config:\n%s", cfg)

    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    all_results = {}
    for dataset_name in cfg.DATASETS.TESTS:
        data_loader, num_query = build_reid_test_loader(cfg, dataset_name=dataset_name)
        outputs = run_inference(model, data_loader, flip_test=args.flip_test or cfg.TEST.FLIP.ENABLED)
        result = compute_metrics(
            cfg,
            outputs["features"],
            outputs["pids"],
            outputs["camids"],
            num_query,
        )

        query_paths = outputs["img_paths"][:num_query]
        gallery_paths = outputs["img_paths"][num_query:]
        use_csg = "CSG" in dataset_name
        rank_lists, aps = compute_rank_lists(
            result["dist"],
            result["query_pids"],
            result["gallery_pids"],
            result["query_camids"],
            result["gallery_camids"],
            use_csg=use_csg,
        )

        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        PathManager.mkdirs(dataset_output_dir)
        save_visualizations(
            dataset_output_dir,
            rank_lists,
            aps,
            query_paths,
            gallery_paths,
            result["dist"],
            args.num_vis,
            args.vis_topk,
            args.vis_rank_sort,
        )

        np.savez_compressed(
            os.path.join(dataset_output_dir, "test_features.npz"),
            query_features=result["query_features"].cpu().numpy(),
            gallery_features=result["gallery_features"].cpu().numpy(),
            query_pids=result["query_pids"],
            gallery_pids=result["gallery_pids"],
            query_camids=result["query_camids"],
            gallery_camids=result["gallery_camids"],
            dist=result["dist"],
        )
        with open(os.path.join(dataset_output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(result["metrics"], handle, indent=2)

        logger.info("Results for %s: %s", dataset_name, result["metrics"])
        all_results[dataset_name] = result["metrics"]

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)


if __name__ == "__main__":
    main(default_argument_parser().parse_args())
