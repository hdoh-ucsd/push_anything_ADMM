# Obstacle weight selection

No live obstacle weights were selected. Phase A synchronized forensic replays
were not launched because C3+ has no controlled experiment seed, and the
initial-C3 transaction bootstrap in logger v2 has not yet passed a runtime
smoke. Consequently there is no synchronized `w_prevent_star` distribution
from which a scientifically defensible live grid could be derived.

The scenario YAML value `obstacle_cost_weight: 5000.0` belongs to the legacy
exponential ranking potential. Under the current scene launcher,
`SAMPLING_C3_OBSTACLE_MODE=lcs_contact` makes the live ranking obstacle scalar
zero unless the experimental `relu_footprint` rank mode and a positive
`SAMPLING_C3_OBS_RELU_W` are explicitly selected. Therefore `5000` was not
silently relabeled as a current ReLU weight.
