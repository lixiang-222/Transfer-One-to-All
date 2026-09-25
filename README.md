# Transfer One to All

## Configuration

- Qwen3-4B rubric generator and policy
- Frozen Qwen3-8B judge
- LoRA rank 16
- `K=4`, `G=4`, `lambda=0.1`
- Learning rate `1e-6`, KL coefficient `0.001`
- 10 alternating stages, 20 update steps per stage
- 64 training prompts and 64 validation prompts per update

## Run

```bash
python -m pip install -e .
transferone-demo
python -m unittest discover -s tests -v
```
