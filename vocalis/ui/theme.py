"""Dark studio theme. Both color modes use the same palette."""

from __future__ import annotations

import gradio as gr


def studio_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.zinc,
        neutral_hue=gr.themes.colors.zinc,
        font=("IBM Plex Sans", "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=("IBM Plex Mono", "ui-monospace", "Consolas", "monospace"),
    ).set(
        body_background_fill="#101114",
        body_background_fill_dark="#101114",
        body_text_color="#e6e8ee",
        body_text_color_dark="#e6e8ee",
        body_text_color_subdued="#a3a8b3",
        body_text_color_subdued_dark="#a3a8b3",
        background_fill_primary="#181b21",
        background_fill_primary_dark="#181b21",
        background_fill_secondary="#14171c",
        background_fill_secondary_dark="#14171c",
        block_background_fill="#181b21",
        block_background_fill_dark="#181b21",
        block_border_color="#2c313a",
        block_border_color_dark="#2c313a",
        block_label_text_color="#d5d8de",
        block_label_text_color_dark="#d5d8de",
        block_title_text_color="#e6e8ee",
        block_title_text_color_dark="#e6e8ee",
        input_background_fill="#12151a",
        input_background_fill_dark="#12151a",
        input_border_color="#343a46",
        input_border_color_dark="#343a46",
        button_primary_background_fill="#1f8a70",
        button_primary_background_fill_dark="#1f8a70",
        button_primary_background_fill_hover="#18745e",
        button_primary_background_fill_hover_dark="#18745e",
        button_primary_text_color="#f4f7f6",
        button_primary_text_color_dark="#f4f7f6",
        button_secondary_background_fill="#22262e",
        button_secondary_background_fill_dark="#22262e",
        button_secondary_text_color="#e6e8ee",
        button_secondary_text_color_dark="#e6e8ee",
        border_color_accent="#1f8a70",
        border_color_accent_dark="#1f8a70",
        shadow_drop="none",
    )
