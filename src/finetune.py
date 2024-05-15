import hydra
from omegaconf import DictConfig, OmegaConf
from EventStream.transformer.lightning_modules.fine_tuning import FinetuneConfig, train
from pathlib import Path
from EventStream.data.config import PytorchDatasetConfig

@hydra.main(version_base=None, config_path=".", config_name="finetune_config")
def main(cfg: DictConfig, data_config_path: str = None):
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    if data_config_path:
        data_config_fp = Path(data_config_path)
        print(f"Loading data_config from {data_config_fp}")
        reloaded_data_config = PytorchDatasetConfig.from_json_file(data_config_fp)
        cfg = FinetuneConfig(**cfg, data_config=reloaded_data_config)
    else:
        cfg = FinetuneConfig(**cfg)

    print("Loaded FinetuneConfig:")
    print(cfg)

    try:
        tuning_metrics, held_out_metrics = train(cfg)
    except Exception as e:
        print(f"Error during training: {e}")
    else:
        print("Training completed successfully.")
        print(f"Tuning metrics: {tuning_metrics}")
        print(f"Held-out metrics: {held_out_metrics}")

if __name__ == "__main__":
    main()