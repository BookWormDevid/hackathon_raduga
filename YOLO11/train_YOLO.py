import torch
from ultralytics import YOLO
import yaml
import os
import logging
from datetime import datetime
import random
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_environment():
    """Configure the execution environment with proper checks"""
    # Memory optimization
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Environment verification
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        for i in range(device_count):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")

        # Check available memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        logger.info(f"GPU memory allocated: {allocated / 1024 ** 2:.2f} MB")
        logger.info(f"GPU memory reserved: {reserved / 1024 ** 2:.2f} MB")


def verify_data_config(config_path='data.yaml'):
    """Validate the data configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            logger.info("Data configuration contents:")
            logger.info(data)

            required_keys = ['train', 'val', 'names', 'nc']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key in data.yaml: {key}")

                if key in ['train', 'val']:
                    path = os.path.abspath(data[key])
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Path not found: {path}")

                    # Verify directory contains images
                    if not any(fname.lower().endswith(('.png', '.jpg', '.jpeg')) for fname in os.listdir(path)):
                        logger.warning(f"No images found in {path}")

                    logger.info(f"{key} path exists ({path})")

            logger.info(f"Dataset contains {data['nc']} classes: {data['names']}")
            return data

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error verifying data config: {e}")
        raise


def get_train_config(data_config_path, experiment_name=None):
    """Create training configuration with sensible defaults"""
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = {
        'data': os.path.abspath(data_config_path),
        'epochs': 100,
        'batch': 8 if torch.cuda.is_available() else 4,
        'imgsz': 640,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'name': experiment_name,
        'single_cls': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'patience': 20,
        'plots': True,
        'save': True,
        'save_period': 10,
        'amp': True,  # Automatic Mixed Precision
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'weight_decay': 0.0005,
        'box': 7.5,  # box loss gain
        'cls': 0.5,  # cls loss gain
        'dfl': 1.5,  # dfl loss gain
        'close_mosaic': 10,  # disable mosaic for last epochs
        'resume': False,
        'workers': 8 if torch.cuda.is_available() else 4,
        'seed': SEED
    }

    # Enable multi-GPU if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        config['device'] = ','.join(str(i) for i in range(torch.cuda.device_count()))
        logger.info(f"Using multiple GPUs: {config['device']}")

    return config


def load_model(model_path='yolo11s.pt'):
    """Load and verify the pretrained model"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)

        # Verify model structure
        logger.info(f"Model architecture: {model.model}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def freeze_model_layers(model, freeze_pattern='backbone'):
    """Freeze model layers based on pattern"""
    try:
        total_params = 0
        frozen_params = 0

        if freeze_pattern == 'backbone':
            # Freeze backbone layers (more precise than arbitrary number)
            for name, param in model.model.named_parameters():
                if 'model.0.' in name or 'model.1.' in name or 'model.2.' in name or 'model.3.' in name:
                    param.requires_grad = False
                    frozen_params += param.numel()
                total_params += param.numel()
        elif freeze_pattern == 'partial':
            # Freeze first 75% of layers
            all_params = list(model.model.named_parameters())
            freeze_up_to = int(len(all_params) * 0.75)
            for i, (name, param) in enumerate(all_params):
                if i < freeze_up_to:
                    param.requires_grad = False
                    frozen_params += param.numel()
                total_params += param.numel()

        logger.info("\nLayer freezing summary:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,} ({frozen_params / total_params:.1%})")
        logger.info(
            f"Trainable parameters: {total_params - frozen_params:,} "
            f"({(total_params - frozen_params) / total_params:.1%})")

        # Log trainable parameters
        logger.info("\nTrainable parameters:")
        for name, param in model.model.named_parameters():
            if param.requires_grad:
                logger.info(f"- {name}")

        return model

    except Exception as e:
        logger.error(f"Error freezing layers: {e}")
        raise


def train_model(model, config):
    """Execute model training with proper resource management"""
    try:
        logger.info("\nStarting training with configuration:")
        for k, v in config.items():
            logger.info(f"- {k}: {v}")

        # Train the model
        results = model.train(**config)

        # Validate the model
        metrics = model.val()
        logger.info("\nValidation metrics:")
        logger.info(f"- mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"- Precision: {metrics.box.p:.4f}")
        logger.info(f"- Recall: {metrics.box.r:.4f}")

        # Save the trained model
        best_model_path = os.path.join('runs', 'detect', config['name'], 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            logger.info(f"\nSaved best model to: {best_model_path}")
        else:
            logger.warning("Best model not found in expected location")

        return results, metrics

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Attempt to save interrupted training
        if 'model' in locals():
            try:
                interrupt_path = f"interrupted_{config['name']}.pt"
                model.save(interrupt_path)
                logger.info(f"Saved interrupted model to {interrupt_path}")
            except Exception as save_error:
                logger.error(f"Failed to save interrupted model: {save_error}")
        raise


def main():
    """Main execution function"""
    try:
        logger.info("Starting training process")

        # 1. Setup environment
        setup_environment()

        # 2. Verify data configuration
        data_config = verify_data_config()

        # 3. Prepare training configuration
        train_config = get_train_config(
            data_config_path='data.yaml',
            experiment_name='Search_rescue_YOLO11n'
        )

        # 4. Load model
        model = load_model()

        # 5. Freeze layers
        #  model = freeze_model_layers(model, freeze_pattern='backbone')

        # 6. Train model
        train_results, val_metrics = train_model(model, train_config)

        logger.info("\nTraining completed successfully!")
        return train_results, val_metrics

    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        logger.info("Execution completed. GPU cache cleared.")


if __name__ == '__main__':
    main()
