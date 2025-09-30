# Copilot Instructions for WiFo

## Project Overview
WiFo is a Wireless Foundation Model for Channel Prediction. The project includes pre-trained models for inference and supports multi-dataset unified learning and zero-shot generalization. Key components include data loading, embedding, model definition, training, and utilities.

## Codebase Structure
- `src/`: Contains the main source code.
  - `main.py`: Entry point for running experiments.
  - `DataLoader.py`: Handles dataset loading and preprocessing.
  - `Embed.py`: Defines embedding strategies.
  - `mask_strategy.py`: Implements masking strategies for training.
  - `model.py`: Defines the model architecture, including the `WiFo` model and its components.
  - `train.py`: Contains training logic.
  - `utils.py`: Utility functions for common tasks.
- `experiments/`: Stores experiment results, including predictions and model checkpoints.
- `logs/`: Contains logs for different experiments.
- `weights/`: Pre-trained model weights.

## Developer Workflows
### Running Inference
Use `main.py` to run inference. Example commands:
```bash
python src\main.py --device_id 0 --size tiny --mask_strategy_random none --mask_strategy temporal --dataset D17 --file_load_path ./weights/wifo_tiny --few_ratio 0.0 --t_patch_size 4 --patch_size 4 --batch_size 128 --pos_emb SinCos_3D
```

### Training
Modify `train.py` to implement training workflows. Ensure datasets are prepared and pre-trained weights are available in the `weights/` directory.

### Debugging
- Use `logs/` to analyze experiment logs.
- Check `result.txt` and `result_all.txt` in `experiments/` for evaluation outputs.

## Project-Specific Conventions
- **Masking Strategies**: Defined in `mask_strategy.py`. Examples include `temporal`, `fre`, and `random`. These strategies are used in the `WiFo` model to mask input data during training and inference.
- **Dataset Naming**: Datasets are referred to as `D1`, `D2`, ..., `D17`.
- **Model Sizes**: Supported sizes include `tiny`, `base`, etc., as defined in the `WiFo_model` function in `model.py`.
- **Positional Embedding**: Use `SinCos_3D` for positional embeddings. The `WiFo` model supports multiple embedding strategies, including `trivial` and `SinCos`.

## Key Patterns and Examples
- **Model Definition**: The `WiFo` model in `model.py` is a masked autoencoder with a Vision Transformer backbone. It includes encoder and decoder blocks, masking strategies, and positional embeddings. Example:
  ```python
  model = WiFo(
      embed_dim=64,
      depth=6,
      decoder_embed_dim=64,
      decoder_depth=4,
      num_heads=8,
      ...
  )
  ```
- **Masking Strategies**: Implemented in `mask_strategy.py` and used in the `forward_encoder` method of the `WiFo` model. Example:
  ```python
  x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T)
  ```
- **Positional Embeddings**: Defined in `Embed.py` and used in the `WiFo` model. Example:
  ```python
  pos_embed = get_2d_sincos_pos_embed(...)
  ```

## External Dependencies
- Python 3.9
- PyTorch 2.0.0
- NVIDIA GPU with CUDA
- Additional Python packages listed in `requirements.txt`.

## Notes for AI Agents
- Follow the structure and conventions outlined above.
- Ensure compatibility with the specified Python and PyTorch versions.
- Validate changes by running inference or training workflows.
- Use `logs/` and `experiments/` for debugging and evaluation.
- When modifying `model.py`, ensure that changes align with the `WiFo` model's architecture and its integration with masking strategies and positional embeddings.

For further details, refer to the `README.md` file.