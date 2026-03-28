from .vae_dataset import ValDataset
from torch.utils.data import DataLoader


def build_val_loader(config):
    val_dataset = ValDataset(
        config["val_file"],
        text_file=config["text_file"],
        writer_file=config.get("writer_file"),
        transform=None,
    )

    config["num_text_embedding"] = len(val_dataset.text_cache) + 1  # 0 for padding

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("val_batch_size", 50),
        shuffle=True,
        num_workers=4,
        collate_fn=ValDataset.collate_fn,
    )

    return val_loader, config
