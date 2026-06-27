# Radial Wavefunction Polars Plot Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the wavefunction plotting functions with Polars-first APIs that compute intermediate values using Polars expressions and only pass Polars Series to Matplotlib at the plotting boundary.

**Architecture:** `rwfn_plot` handles one `pl.DataFrame`; `rwfns_compare_plot` handles multiple `pl.DataFrame` objects with labels, colors, and line styles. Both functions share helpers for orbital column validation, x-axis transforms, y-series construction, auto x-limit detection, subplot layout, and axis finalization.

**Tech Stack:** Python 3.14, Polars, Matplotlib, pytest, Ruff.

---

### Task 1: Define The New Public Behavior

**Files:**
- Modify: `tests/test_plot_functions.py`

- [ ] **Step 1: Replace old wavefunction tests with new API tests**

Write tests that import `rwfn_plot` and `rwfns_compare_plot` from `graspkit.utils.plot_functions`.

Required behaviors:
- `rwfn_plot(data=df, orbitals=["1s "])` defaults to `plot_mode="density"` and plots `P(1s ) ** 2 + Q(1s ) ** 2`.
- `rwfn_plot(..., plot_mode="P")` plots only the `P(...)` column.
- `rwfn_plot(..., plot_mode="Q")` plots only the `Q(...)` column.
- `rwfn_plot(..., plot_mode="components")` lays out `P` above `Q` for each orbital.
- `rwfns_compare_plot(data_list=[df_old, df_new], labels=["old", "new"], ...)` overlays one line per DataFrame in each subplot.
- `x_transform="sqrt"` uses Polars `.sqrt()` and produces the same plotted x values as `sqrt(r)`.
- `x_transform="linear"` keeps raw x values and defaults to a linear Matplotlib axis.
- `auto_max_x=True` computes the x limit from the last value above `x_tail_threshold` using Polars expressions.
- Invalid `plot_mode`, missing columns, invalid `x_tail_threshold`, and invalid `x_tail_padding` raise clear `ValueError`s.

- [ ] **Step 2: Run the new tests before implementation**

Run:

```bash
uv run pytest tests/test_plot_functions.py -q
```

Expected: FAIL during import because `rwfn_plot` and `rwfns_compare_plot` do not exist yet.

### Task 2: Add Polars-First Helpers

**Files:**
- Modify: `src/graspkit/utils/plot_functions.py`

- [ ] **Step 1: Add imports**

Add:

```python
from typing import Any, Literal
import polars as pl
```

Define:

```python
PlotMode = Literal["density", "P", "Q", "components"]
XTransform = Literal["sqrt", "linear", "raw", "log", "ln", "log10"]
```

- [ ] **Step 2: Add orbital and layout helpers**

Implement helpers:

```python
def _component_column(orbital: str, component: Literal["P", "Q"]) -> str:
    return f"{component}({orbital})"

def _validate_rwfn_columns(data: pl.DataFrame, orbitals: list[str], x_col: str) -> None:
    missing_columns = [x_col] if x_col not in data.columns else []
    for orbital in orbitals:
        for component in ("P", "Q"):
            col_name = _component_column(orbital, component)
            if col_name not in data.columns:
                missing_columns.append(col_name)
    if missing_columns:
        raise ValueError(f"Wavefunction DataFrame is missing required columns: {missing_columns}")
```

Implement automatic layout:

```python
def _auto_rwfn_layout(n_items: int) -> str:
    ncols = int(np.ceil(np.sqrt(n_items)))
    nrows = int(np.ceil(n_items / ncols))
    return f"{nrows}x{ncols}"
```

For `plot_mode="components"`, default layout is `2x{len(orbitals)}`.

- [ ] **Step 3: Add Polars series helpers**

Implement `_x_series` using Polars expressions:

```python
def _x_series(data: pl.DataFrame, x_col: str, x_transform: XTransform) -> pl.Series:
    expr = pl.col(x_col)
    if x_transform == "sqrt":
        expr = expr.sqrt()
    elif x_transform in {"linear", "raw"}:
        expr = expr
    elif x_transform in {"log", "ln"}:
        expr = expr.log()
    elif x_transform == "log10":
        expr = expr.log10()
    else:
        raise ValueError(...)
    series = data.select(expr.alias(x_col)).to_series()
    if series.is_nan().any() or series.is_infinite().any():
        raise ValueError(...)
    return series
```

Implement `_rwfn_y_series`:

```python
def _rwfn_y_series(data: pl.DataFrame, orbital: str, plot_mode: Literal["density", "P", "Q"]) -> pl.Series:
    p_col = _component_column(orbital, "P")
    q_col = _component_column(orbital, "Q")
    if plot_mode == "density":
        return data.select((pl.col(p_col) ** 2 + pl.col(q_col) ** 2).alias(orbital)).to_series()
    if plot_mode == "P":
        return data[p_col]
    if plot_mode == "Q":
        return data[q_col]
    raise ValueError(...)
```

- [ ] **Step 4: Add auto max x helper**

Implement `_last_significant_x_from_polars` that filters rows using Polars:

```python
def _last_significant_x_from_polars(
    data: pl.DataFrame,
    x_col: str,
    orbitals: list[str],
    plot_mode: PlotMode,
    x_transform: XTransform,
    threshold: float,
) -> float | None:
    masks = []
    for orbital in orbitals:
        if plot_mode == "density":
            masks.append((pl.col(_component_column(orbital, "P")) ** 2 + pl.col(_component_column(orbital, "Q")) ** 2).abs() > threshold)
        elif plot_mode == "P":
            masks.append(pl.col(_component_column(orbital, "P")).abs() > threshold)
        elif plot_mode == "Q":
            masks.append(pl.col(_component_column(orbital, "Q")).abs() > threshold)
        elif plot_mode == "components":
            masks.append(pl.col(_component_column(orbital, "P")).abs() > threshold)
            masks.append(pl.col(_component_column(orbital, "Q")).abs() > threshold)
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask | mask
    result = data.with_columns(_x_expr(x_col, x_transform).alias("__x_plot")).filter(combined_mask).select(pl.col("__x_plot").last())
    value = result.item()
    return None if value is None else float(value)
```

### Task 3: Implement Shared Plot Core

**Files:**
- Modify: `src/graspkit/utils/plot_functions.py`

- [ ] **Step 1: Build plot items**

For `density`, `P`, and `Q`, plot items are one item per orbital:

```python
[("density", orbital), ...]
[("P", orbital), ...]
[("Q", orbital), ...]
```

For `components`, plot items are P row followed by Q row:

```python
[("P", orbital_1), ("P", orbital_2), ..., ("Q", orbital_1), ("Q", orbital_2), ...]
```

- [ ] **Step 2: Share axis formatting**

Keep `_resolve_wavefunction_xscale`, `_use_plain_linear_x_axis`, `_get_wavefunction_colors`, `_get_wavefunction_linestyles`, and `_finalize_wavefunction_layout`.

Apply axis settings after plotting each subplot:
- `set_xlim(0, max_x)`
- `symlog` with `linthresh` when selected
- `log` when selected
- plain tick labels for linear/raw transforms on a linear axis
- legend only when labels are present

### Task 4: Implement `rwfn_plot`

**Files:**
- Modify: `src/graspkit/utils/plot_functions.py`

- [ ] **Step 1: Add function signature**

```python
def rwfn_plot(
    data: pl.DataFrame,
    orbitals: list[str],
    x_col: str = "r(a.u)",
    plot_mode: PlotMode = "density",
    layout: str | None = None,
    alpha: float = 0.75,
    max_x: float | None = None,
    xscale: str | None = None,
    linthresh: int = 1,
    suptitle: str = "Radial Wavefunction",
    color: str | None = None,
    linestyle: Any = "-",
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_transform: XTransform = "sqrt",
    auto_max_x: bool = True,
    x_tail_threshold: float = 0.0,
    x_tail_padding: float = 1.05,
) -> tuple[Figure, np.ndarray]:
```

- [ ] **Step 2: Implement behavior**

Use `_x_series` once per DataFrame.

For each plot item:
- use `_rwfn_y_series` for `density`, `P`, and `Q`
- use `_rwfn_y_series` with the item component for `components`
- call `axes[row, col].plot(x_values, y_values, ...)`

### Task 5: Implement `rwfns_compare_plot`

**Files:**
- Modify: `src/graspkit/utils/plot_functions.py`

- [ ] **Step 1: Add function signature**

```python
def rwfns_compare_plot(
    data_list: list[pl.DataFrame],
    orbitals: list[str],
    x_col: str = "r(a.u)",
    labels: list[str] | None = None,
    plot_mode: PlotMode = "density",
    layout: str | None = None,
    alpha: float = 0.75,
    max_x: float | None = None,
    xscale: str | None = None,
    linthresh: int = 1,
    suptitle: str = "Radial Wavefunction Comparison",
    colors: list[str] | None = None,
    linestyles: list[Any] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_transform: XTransform = "sqrt",
    auto_max_x: bool = True,
    x_tail_threshold: float = 0.0,
    x_tail_padding: float = 1.05,
) -> tuple[Figure, np.ndarray]:
```

- [ ] **Step 2: Implement behavior**

Validate every DataFrame.

For each subplot, overlay one line per DataFrame:

```python
for data_index, data in enumerate(data_list):
    x_values = _x_series(data, x_col, x_transform)
    y_values = _rwfn_y_series(data, orbital, y_mode)
    ax.plot(x_values, y_values, label=labels[data_index], color=colors[data_index], linestyle=linestyles[data_index])
```

### Task 6: Remove Old Wavefunction APIs

**Files:**
- Modify: `src/graspkit/utils/plot_functions.py`

- [ ] **Step 1: Delete old functions**

Remove:
- `plot_wavefunction`
- `auto_plot_wavefunction_comparison`
- `_column_to_numpy`
- `_transform_x_values`
- `_last_significant_x_value`
- `_calculate_default_max_x`
- `_calculate_auto_max_x`
- `_calculate_auto_max_x_from_arrays`
- `_order_wavefunction_columns_for_layout`

Keep NumPy import if `inter_coupling_channel_bar` still uses it.

### Task 7: Verify

**Files:**
- Modify: `tests/test_plot_functions.py`
- Modify: `src/graspkit/utils/plot_functions.py`

- [ ] **Step 1: Run focused tests**

```bash
uv run pytest tests/test_plot_functions.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run Ruff on touched files**

```bash
uv run ruff check src/graspkit/utils/plot_functions.py tests/test_plot_functions.py
```

Expected: no Ruff errors.

- [ ] **Step 3: Run related import tests**

```bash
uv run pytest tests/test_import_boundaries.py tests/test_logical_packages.py -q
```

Expected: all tests pass and plotting remains lazily imported.
