# stochastic-process-simulation

Simulation and LaTeX report for Brownian motion, geometric Brownian motion, the Ornstein--Uhlenbeck process, and the Vasicek interest-rate model.

## Reproduce

```bash
uv run python -m unittest -v test_simulation.py
uv run python generate_data.py
latexmk -pdf -output-directory=figures figures/paths.tex
latexmk -pdf -output-directory=figures figures/distributions.tex
latexmk -pdf main.tex
```

The submission-ready report is `Harsh_Gupta_22IM3FP29_Financial_Mathematics_Project.pdf`.
