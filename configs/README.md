# Configuration Guide

## Overview

Configuration files define experiments, datasets, and execution rules for the automated fine-tuning system.

## File Format

Use JSON format (`.json` extension). For reference with comments, see `config_example_with_comments.jsonc`.

## Structure

```json
{
  "dataset": { ... },
  "output_dir": "results",
  "system_prompt": "...",
  "experiments": { ... }
}
```

## Rules System

### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `>` | Greater than | `"exp1.f1" > "exp2.f1"` |
| `<` | Less than | `"exp1.latency" < "exp2.latency"` |
| `>=` | Greater than or equal | `"exp1.f1" >= 0.8` |
| `<=` | Less than or equal | `"exp2.loss" <= 0.5` |
| `==` | Equal to | `"exp1.epochs" == 2` |
| `!=` | Not equal to | `"exp1.f1" != "exp2.f1"` |

### Available Metrics

Format: `"experiment_name.metric_name"`

**Evaluation Metrics:**
- `f1` - F1 score
- `precision` - Precision score
- `recall` - Recall score
- `latency` - Average inference latency

**Training Metrics:**
- `last_train_loss` - Final training loss
- `min_train_loss` - Minimum training loss achieved
- `last_eval_loss` - Final evaluation loss
- `min_eval_loss` - Minimum evaluation loss achieved

### Rule Examples

#### Example 1: Run if model hasn't converged
```json
"rules": [
  {
    "conditions": [
      { "left": "exp1.last_train_loss", "op": ">", "right": "exp1.min_train_loss" }
    ]
  }
]
```

#### Example 2: Run if exp1 better than exp2
```json
"rules": [
  {
    "conditions": [
      { "left": "exp1.f1", "op": ">", "right": "exp2.f1" }
    ]
  }
]
```

#### Example 3: Multiple conditions (ALL must be true)
```json
"rules": [
  {
    "conditions": [
      { "left": "exp1.f1", "op": ">", "right": "exp2.f1" },
      { "left": "exp2.last_eval_loss", "op": ">", "right": "exp2.min_eval_loss" },
      { "left": "exp2.last_train_loss", "op": "<=", "right": "exp2.min_train_loss" }
    ]
  }
]
```

#### Example 4: Run if overfitting detected
```json
"rules": [
  {
    "conditions": [
      { "left": "exp1.last_eval_loss", "op": ">", "right": "exp1.min_eval_loss" },
      { "left": "exp1.last_train_loss", "op": "<=", "right": "exp1.min_train_loss" }
    ]
  }
]
```

## Execution Logic

### run_always: true
- Experiment always executes
- Rules are ignored
- Use for baseline experiments

### run_always: false
- Experiment executes ONLY if ALL conditions are TRUE
- If any condition fails, experiment is skipped
- Use for conditional experiments

### Empty rules: []
- Same as `run_always: true`
- Experiment always executes

## Common Patterns

### Pattern 1: Baseline + Conditional
```json
"exp1": {
  "run_always": true,
  "rules": []
},
"exp2": {
  "run_always": false,
  "rules": [
    {
      "conditions": [
        { "left": "exp1.f1", "op": "<", "right": "0.7" }
      ]
    }
  ]
}
```

### Pattern 2: Sequential Experiments
```json
"exp1": { "run_always": true, "rules": [] },
"exp2": { "run_always": true, "rules": [] },
"exp3": {
  "run_always": false,
  "rules": [
    {
      "conditions": [
        { "left": "exp1.f1", "op": ">", "right": "exp2.f1" }
      ]
    }
  ]
}
```

### Pattern 3: Convergence Check
```json
"exp2": {
  "run_always": false,
  "rules": [
    {
      "conditions": [
        { "left": "exp1.last_train_loss", "op": ">", "right": "exp1.min_train_loss" }
      ]
    }
  ]
}
```

## Tips

1. **Start Simple**: Use `run_always: true` for initial experiments
2. **Test Rules**: Verify conditions with small datasets first
3. **Use Metrics**: Reference previous experiment metrics for decisions
4. **Document Logic**: Add comments in separate `.jsonc` file for reference
5. **Check Order**: Experiments execute in order, ensure dependencies are met

## Validation

Before running, validate your config:

```python
import json
with open('config.json') as f:
    config = json.load(f)
print("✅ Config is valid JSON")
```

## Examples

See:
- `config_gemma3.json` - Simple configuration
- `config_qwen3.json` - Advanced with rules
- `config_example_with_comments.jsonc` - Fully commented reference
