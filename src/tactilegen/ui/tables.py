"""Reusable saved-items tables for the TactileGen Gradio UI."""

from collections.abc import Callable
from copy import copy
from typing import Any

import gradio as gr

# Column spec for table headers/cells: (label, scale, min_width).
ColumnSpec = tuple[str, int, int]

# =============================================================================
# Small reusable building blocks
# =============================================================================


def _render_items_table_header(columns: list[ColumnSpec]) -> None:
    """Render the centered header row of a saved-items table."""
    with gr.Row(elem_classes=["center", "px-8px"]):
        for label, scale, min_width in columns:
            with gr.Column(scale=scale, min_width=min_width, elem_classes="center"):
                gr.Markdown(f"<center><strong>{label}</strong></center>")


def table_cell_image(value: Any) -> gr.Image:
    """Thumbnail-style `gr.Image` used in table cells."""
    return gr.Image(
        value=value,
        height=64,
        show_label=False,
        buttons=["fullscreen"],
        container=False,
    )


def cell_column(scale: int, min_width: int, build: Callable[[], Any]) -> None:
    """Centered column wrapper used inside `render_content_cells` callbacks."""
    with gr.Column(scale=scale, min_width=min_width, elem_classes="center"):
        build()


# =============================================================================
# Generic items-table backbone
# =============================================================================


def items_table(
    *,
    items_state: gr.State,
    empty_message: str,
    columns: list[ColumnSpec],
    render_content_cells: Callable[[Any], None],
) -> None:
    """Render a saved-items table containing as rows the items of items_state.

    gr.render automatically updates the table when items_state changes.
    """

    @gr.render(inputs=[items_state])
    def _render(items: list) -> None:
        if not items:
            gr.Markdown(empty_message)
            return
        with gr.Group():
            _render_items_table_header(
                columns + [("Enabled", 1, 28), ("Delete", 1, 28)]
            )
            for i, item in enumerate(items):
                _items_table_row(
                    i,
                    item,
                    items_state,
                    render_content_cells,
                )


def _items_table_row(
    idx: int,
    item: Any,
    items_state: gr.State,
    render_content_cells: Callable[[Any], None],
) -> None:
    with gr.Row(variant="compact", elem_classes="center"):
        render_content_cells(item)
        with gr.Column(scale=1, min_width=28, elem_classes="center"):
            enabled_chk = gr.Checkbox(
                value=item.enabled,
                label="Enabled",
                show_label=False,
                interactive=True,
                container=False,
            )
        with gr.Column(scale=1, min_width=28, elem_classes="center"):
            del_btn = gr.Button("✕", size="md", variant="stop", min_width=28)

    def toggle(enabled: bool, items: list) -> list | tuple:
        # Need to replace items[idx] with a fresh copy so items_state.change() triggers.
        items[idx] = copy(items[idx])
        items[idx].enabled = enabled
        return items

    def delete(items: list) -> list | tuple:
        del items[idx]
        return items

    enabled_chk.change(
        fn=toggle, inputs=[enabled_chk, items_state], outputs=[items_state]
    )
    del_btn.click(fn=delete, inputs=[items_state], outputs=[items_state])
