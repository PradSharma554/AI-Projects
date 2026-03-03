---
title: Sample Documentation
author: Test Author
---

# Getting Started

This is the getting started guide for Ragged.

## Installation

Install Ragged using pip:

```bash
pip install ragged
```

## Configuration

Create a `config.yaml` file in your project root:

```yaml
ollama:
  url: "http://localhost:11434"
  model: "llama3.2"
```

# Advanced Usage

## Custom Parsers

You can create custom parsers by implementing the `BaseParser` protocol.

## Hybrid Retrieval

Ragged uses BM25 + vector search with Reciprocal Rank Fusion.
