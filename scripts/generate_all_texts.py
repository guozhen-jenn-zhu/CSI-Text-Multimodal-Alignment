"""Generate text pair registries for all datasets and tasks."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "csibench": "configs/dataset/csibench.yaml",
    "xrf55": "configs/dataset/xrf55.yaml",
    "signfi": "configs/dataset/signfi.yaml",
}


def generate_for_task(
    dataset_name: str,
    task_name: str,
    task_root: str,
    h5_key: str,
    output_dir: str,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    num_paraphrases: int = 5,
) -> Path:
    """Generate a pair registry for a single task."""
    from data.metadata_loader import MetadataLoader
    from data.text_generation.generate_texts import TextGenerator

    loader = MetadataLoader(task_root=task_root, h5_key=h5_key)
    all_ids = loader.get_all_sample_ids()
    sample_labels = {sid: loader.get_label(sid) for sid in all_ids}

    sample_metadata = None
    if dataset_name == "csibench":
        sample_metadata = {}
        for sid in all_ids:
            meta = loader.get_metadata(sid)
            if all(k in meta for k in ("user", "activity", "distance")):
                sample_metadata[sid] = {
                    "user_id": str(meta["user"]),
                    "activity": str(meta["activity"]),
                    "proximity": str(meta["distance"]),
                }

    gen = TextGenerator(
        dataset_name=dataset_name,
        use_llm_paraphrase=use_llm,
        llm_model=llm_model,
        num_paraphrases=num_paraphrases,
    )

    out_path = Path(output_dir) / dataset_name / task_name / "pair_registry.json"
    gen.generate_all(
        sample_ids=all_ids,
        sample_labels=sample_labels,
        sample_metadata=sample_metadata,
        output_path=str(out_path),
    )

    logger.info("Generated: %s (%d samples)", out_path, len(all_ids))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text registries for all datasets")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--output_dir", type=str, default="data/generated_texts")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num_paraphrases", type=int, default=5)
    parser.add_argument("--s3_sync", action="store_true", help="Upload results to S3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    generated = []

    for dataset_name in args.datasets:
        config_path = DATASET_CONFIGS[dataset_name]
        if not Path(config_path).is_file():
            logger.warning("Config not found: %s -- skipping %s", config_path, dataset_name)
            continue

        with open(config_path) as f:
            ds_cfg = yaml.safe_load(f)

        data_root = ds_cfg.get("data_root", "/data/fm_downstream")
        tasks = ds_cfg.get("tasks", {})

        if isinstance(tasks, dict):
            for task_name, task_cfg in tasks.items():
                folder = task_cfg.get("folder", task_name)
                h5_key = task_cfg.get("h5_key", "CSI_amps")
                task_root = str(Path(data_root) / folder)

                if not Path(task_root).is_dir():
                    logger.warning("Task root not found: %s -- skipping", task_root)
                    continue

                out = generate_for_task(
                    dataset_name=dataset_name,
                    task_name=task_name,
                    task_root=task_root,
                    h5_key=h5_key,
                    output_dir=args.output_dir,
                    use_llm=args.use_llm,
                    llm_model=args.llm_model,
                    num_paraphrases=args.num_paraphrases,
                )
                generated.append(out)

    if args.s3_sync and generated:
        try:
            from utils.s3_sync import S3Syncer
            syncer = S3Syncer(bucket="rnd-sagemaker", prefix="Multimodal_CSI_Text")
            syncer.upload_dataset(args.output_dir)
            logger.info("Synced generated texts to S3")
        except Exception as e:
            logger.warning("S3 sync failed: %s", e)

    print(f"\nGenerated {len(generated)} pair registries under {args.output_dir}/")


if __name__ == "__main__":
    main()
