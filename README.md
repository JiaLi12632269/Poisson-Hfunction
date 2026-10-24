# Poisson h-function Methods

Source code for the FC, RW, EB, and IM initial interval methods and the
corresponding h-function refinement procedures. The repository separates the
four method implementations from the shared refinement engine.

## Initial interval methods

The original R source files are stored directly in `methods/`:

- `FC_initial_interval.R`
- `RW_initial_interval.R`
- `EB_initial_interval.R`
- `IM_initial_interval.R`

The algorithm statements and calculations are preserved. Chinese comments
were translated into English.

## H-function refinement

- `hfunction_torch.py` provides the shared CPU and PyTorch h-function
  calculations, interval refinement, and multi-iteration refinement sequence.
- The R source files and the shared h-function engine are the method
  implementation. Sensitivity-analysis scripts are not included.

## Initial interval workbooks

The four matching workbooks are in `data/` and are read by the h-function
implementation:

- `FC_intervals.xlsx`
- `RW_intervals.xlsx`
- `EB_intervals.xlsx`
- `IM_intervals.xlsx`

Numeric cell values are unchanged. Worksheet names are in English:

- `Initial Interval`
- `Refinement 1`, `Refinement 2`, and so forth

The R scripts print their calculated objects; they do not automatically create
these Excel workbooks. The included workbooks provide the initial tables and
saved refinement levels for the Python implementation.

## Environment

The Python code was used with Python 3.9.13, NumPy 2.0.2, SciPy 1.13.1,
openpyxl 3.1.4, and PyTorch 2.7.1+cu118. Install the Python
dependencies with:

    python -m pip install -r requirements.txt

The R scripts use base R and the `stats` package. For CUDA runs, install a
PyTorch build compatible with the local CUDA environment.

## Running the R methods

Run an initial interval method from the repository root, for example:

    Rscript methods/FC_initial_interval.R

Use the corresponding R file for RW, EB, or IM.

