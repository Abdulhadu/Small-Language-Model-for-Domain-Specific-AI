# Domain-Specific Small Language Model

A custom implementation of a transformer-based language model trained on domain-specific knowledge. This model is designed to understand and generate responses within a specific domain context.

![Training Loss Progress](media/Screenshot%202025-01-31%20162236.png)

## Model Architecture

The model follows a transformer-based architecture with the following key components:
- Multi-head self-attention mechanism
- Feed-forward neural networks
- Layer normalization
- Positional embeddings

### Hyperparameters
- vocab_size = 5000
- context_length = 32
- embedding_dim = 128
- num_heads = 12
- num_layers = 4
- batch_size = 16
- num_steps = 2000


### Dataset Split
- Training dataset size: 4,835 samples
- Validation dataset size: 509 samples

## Key Features

1. **Custom Tokenizer**: Implements BPE (Byte Pair Encoding) tokenization with special tokens
2. **Multi-head Attention**: Uses 12 attention heads for better feature capture
3. **Layer Normalization**: Implements layer normalization for stable training
4. **Position-aware**: Incorporates positional embeddings for sequence awareness

## Model Components

### 1. Attention Mechanism
- Individual attention heads with key, query, and value projections
- Scaled dot-product attention with masking
- Multi-head attention with concatenation and projection

### 2. Feed-Forward Network
- Two-layer neural network with ReLU activation
- Hidden dimension expansion for better representation

### 3. Transformer Block
- Self-attention layer
- Feed-forward network
- Residual connections
- Layer normalization

## Training

The model is trained using:
- Adam optimizer with learning rate 1e-3
- Cross-entropy loss function
- Batch size of 16
- 2000 training steps
- Context length of 32 tokens

## Results

The training shows significant improvement in loss values:
- Initial loss: ~8.7
- Final loss: ~0.04
- Consistent decrease in loss throughout training


## Requirements

- PyTorch
- Tokenizers
- NumPy
- Transformers


## Future Improvements

1. Implement beam search for better text generation
2. Add temperature scaling for generation diversity
3. Implement model checkpointing
4. Add evaluation metrics beyond loss
5. Expand vocabulary size for better coverage


