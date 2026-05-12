import pandas as pd
from pathlib import Path
import pymatviz as pmv
from typing import Optional, Union
import re



def _safe_filename(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def plot_remove_metal_test_mae_ptables(
    results_df: pd.DataFrame,
    *,
    value_col: str = "test_mae",
    group_cols=("data_config", "model_config", "train_config"),
    colorscale: str = "Reds",
    renderer: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
):
    df = results_df.copy()
    df = df[df["split_mode"] == "Remove_Metal"].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    if df.empty:
        return {}
    
    global_min = df[value_col].min()
    global_max = df[value_col].max()
    cscale_range=(global_min, global_max)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figs = {}

    for keys, sub in df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)

        values = (
            sub.dropna(subset=["split_value", value_col])
            .set_index("split_value")[value_col]
            .to_dict()
        )

        title = " | ".join(f"{col}={val}" for col, val in zip(group_cols, keys))

        fig = pmv.ptable_heatmap_plotly(
            values,
            fmt=".3f",
            colorscale=colorscale,
            colorbar={"title": value_col},
            cscale_range=(global_min, global_max),
            show_values=True,
            gap=3,
            font_size=16,
        )

        fig.update_layout(
            title={
                "text": f"Remove-metal {value_col}<br>{title}",
                "x": 0.5,
                "xanchor": "center",
            },
            width=1150,
            height=700,
            margin=dict(l=20, r=20, t=90, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        if renderer is not None:
            fig.show(renderer=renderer)

        if save_dir is not None:
            filename = "__".join(_safe_filename(k) for k in keys) + ".png"
            outpath = save_dir / filename
            fig.write_image(outpath, scale=2)

        figs[keys] = fig

    return figs