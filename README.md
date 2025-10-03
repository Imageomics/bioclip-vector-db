# bioclip-vector-db

This repository will be used to track work for the BioCLIP vector database training data retrieval project as a central location for planning, communication, and implementation.

# Problem Statement
As a user submitting an image for open-ended classification by BioCLIP, I want to know which images from the training dataset BioCLIP considers to be the most similar to my image so I can see the similarities in visible traits among taxa as they are represented in the embedding space.

# Setup

## Environment
```bash
module load cuda/12.4.1
```

## Virtual Environment (uv)
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install project with dependencies
uv pip install -e .
```