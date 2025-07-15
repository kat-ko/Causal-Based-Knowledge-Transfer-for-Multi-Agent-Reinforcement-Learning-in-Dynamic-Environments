# Tables Documentation

This directory contains two types of tables: baseline tables and transfer tables. Each type is generated for different obstacle types (Wall, ReverseU, and U).

## Baseline Tables

Baseline tables (`baseline_*.csv`) contain revisit rates for different agent types in various scenarios. The columns are:

- `Scenario`: The experiment identifier (e.g., "Wall-SS-SE")
- `R2`: Revisit rate for the R2 agent
- `P2`: Revisit rate for the P2 agent
- `P1_CM`: Revisit rate for the P1_CM agent

### Revisit Rate Definition
The revisit rate is calculated as the fraction of state visits where a state was visited more than once during an episode. It is computed as:

```
revisit_rate = (number of states visited > 1) / (total number of state visits)
```

A lower revisit rate indicates more efficient exploration, as the agent is less likely to revisit the same states.

## Transfer Tables

Transfer tables (`transfer_*.csv`) contain revisit rates and their differences for transfer learning scenarios. The columns are:

- `T-ID`: The transfer experiment identifier (e.g., "T-Wall-SS-SE")
- `R2`: Revisit rate for the R2 agent in the learner's scenario
- `P2`: Revisit rate for the P2 agent in the learner's scenario
- `Own CM`: Revisit rate for the learner's own P1_CM agent
- `TRANSFER`: Revisit rate for the transfer agent
- `Δ vs R2`: Difference in revisit rate between TRANSFER and R2 (TRANSFER - R2)
- `Δ vs P2`: Difference in revisit rate between TRANSFER and P2 (TRANSFER - P2)
- `Δ vs Own CM`: Difference in revisit rate between TRANSFER and Own CM (TRANSFER - Own CM)

### Transfer Learning Context
In transfer learning scenarios:
- A teacher agent (P1_CM) learns in one scenario
- A learner agent (TRANSFER) uses the teacher's knowledge to learn in a different scenario
- The differences (Δ) show how much the transfer learning improves over baseline agents

### Interpretation
- Positive Δ values indicate that the transfer agent has a higher revisit rate than the comparison agent
- Negative Δ values indicate that the transfer agent has a lower revisit rate than the comparison agent
- Lower revisit rates are generally better, indicating more efficient exploration

## File Naming Convention
- `baseline_Wall.csv`: Baseline revisit rates for Wall obstacle
- `baseline_ReverseU.csv`: Baseline revisit rates for ReverseU obstacle
- `baseline_U.csv`: Baseline revisit rates for U obstacle
- `transfer_Wall.csv`: Transfer learning revisit rates for Wall obstacle
- `transfer_ReverseU.csv`: Transfer learning revisit rates for ReverseU obstacle
- `transfer_U.csv`: Transfer learning revisit rates for U obstacle 