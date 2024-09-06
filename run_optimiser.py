from models.optimise import FantasyOptimiser

# Initialise optimiser
optimiser = FantasyOptimiser()

# Solve the optimisation problem
optimiser.solve()

# Export solved model, results and summary of actions
model = optimiser.get_model()
results = optimiser.get_results()
summary = optimiser.get_summary()
print(summary)

model.writeLP(
    f"models/fpl_optimiser_{optimiser.team_id}_horizon_{optimiser.horizon}.lp"
)
results.to_csv(
    f"data/results/{optimiser.team_id}_horizon_{optimiser.horizon}.csv", index=False
)
with open(
    f"reports/summary_{optimiser.team_id}_horizon_{optimiser.horizon}.txt", "w"
) as f:
    f.write(summary)
