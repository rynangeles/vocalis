"""Vocalis studio shell.

Two Gradio timers are created when the app is built. Clicks do not create more.

- The job timer runs every 2 seconds only while the current job is queued or running.
- The monitor timer runs every 5 seconds only while the System Monitor page is open.

Each browser session has its own copy of that state. Refreshing a job reads the
existing id and does not submit another generation.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from vocalis.core.config import Settings
from vocalis.core.types import ModelProvider
from vocalis.ui.actions import (
    DeleteOutcome,
    HistoryPage,
    JobView,
    cancel_job,
    delete_history,
    history_detail,
    history_page,
    inspect_job,
    inspect_reference,
    monitor_text,
    settings_text,
    submit_clone,
    submit_design,
)
from vocalis.ui.present import (
    HISTORY_HEADERS,
    HISTORY_PAGE_SIZE,
    PROVIDER_FILTERS,
    STATUS_FILTERS,
    TYPE_FILTERS,
    render_shell,
)
from vocalis.ui.runtime import StudioServices
from vocalis.ui.theme import studio_theme

PAGES = ("design", "clone", "history", "monitor", "settings")
JOB_REFRESH_SECONDS = 2
MONITOR_REFRESH_SECONDS = 5

_MOCK_DESIGN = (
    "Development mode. Generate writes a simulated tone. "
    "It is not VoxCPM2 speech, and this page does not report percentage progress."
)
_MOCK_CLONE = (
    "Development mode. The result is a simulated tone. "
    "Mock mode does not use the reference audio to clone a voice."
)
_PARAMETER_NOTE = (
    "These values are stored with the request. "
    "Mock mode does not apply them to a voice model."
)


def page_visibility(page: str) -> list[bool]:
    return [page == name for name in PAGES]


def button_variants(page: str) -> list[str]:
    return ["primary" if page == name else "secondary" for name in PAGES]


def create_app(settings: Settings, services: StudioServices) -> gr.Blocks:
    """Build the studio. Callbacks delegate to the application services."""
    with gr.Blocks(
        analytics_enabled=False,
        title=f"{settings.app_name} — AI Voice Studio",
        fill_width=True,
    ) as demo:
        job_id = gr.State("")
        history_ids = gr.State([])
        history_offset = gr.State(0)
        selected_history = gr.State("")
        armed_delete = gr.State("")

        with gr.Sidebar(width=280, open=True):
            brand = gr.HTML(render_shell(settings, services.manager))
            nav_design = gr.Button("Voice Design", variant="primary", elem_classes=["vocalis-nav"])
            nav_clone = gr.Button("Voice Cloning", variant="secondary", elem_classes=["vocalis-nav"])
            nav_history = gr.Button("Generation History", variant="secondary", elem_classes=["vocalis-nav"])
            nav_monitor = gr.Button("System Monitor", variant="secondary", elem_classes=["vocalis-nav"])
            nav_settings = gr.Button("Settings", variant="secondary", elem_classes=["vocalis-nav"])

        with gr.Column(visible=True, elem_classes=["vocalis-page"]) as design_page:
            gr.Markdown(_page_intro("Voice Design", settings, _MOCK_DESIGN))
            design_text = gr.Textbox(label="Text to generate", lines=8, placeholder="Write the line to speak.")
            design_description = gr.Textbox(
                label="Voice description",
                lines=3,
                placeholder="Describe the voice. Leave blank to omit it.",
            )
            design_cfg, design_steps, design_normalize, design_denoise = _parameters()
            with gr.Row():
                design_generate = gr.Button("Generate", variant="primary")
                design_reset = gr.Button("Reset form")

        with gr.Column(visible=False, elem_classes=["vocalis-page"]) as clone_page:
            gr.Markdown(_page_intro("Voice Cloning", settings, _MOCK_CLONE))
            clone_text = gr.Textbox(label="Text to generate", lines=6, placeholder="Write the line to speak.")
            clone_file = gr.File(
                label="Reference audio",
                file_count="single",
                file_types=[".wav", ".flac", ".mp3"],
                type="filepath",
            )
            reference_meta = gr.Textbox(label="Reference check", lines=6, interactive=False)
            reference_audio = gr.Audio(label="Reference preview", type="filepath", interactive=False)
            clone_transcript = gr.Textbox(
                label="Reference transcript",
                lines=2,
                placeholder="Optional. Leave blank if you do not have one.",
            )
            clone_cfg, clone_steps, clone_normalize, clone_denoise = _parameters()
            with gr.Row():
                clone_generate = gr.Button("Generate", variant="primary")
                clone_reset = gr.Button("Reset form")

        with gr.Column(visible=False, elem_classes=["vocalis-page"]) as history_page_col:
            gr.Markdown("## Generation History\nNewest generations are loaded one page at a time.")
            with gr.Row():
                type_filter = gr.Dropdown(TYPE_FILTERS, value="Any type", label="Generation type")
                status_filter = gr.Dropdown(STATUS_FILTERS, value="Any status", label="Status")
                provider_filter = gr.Dropdown(PROVIDER_FILTERS, value="Any provider", label="Provider")
            with gr.Row():
                history_prev = gr.Button("Previous page")
                history_next = gr.Button("Next page")
            history_summary = gr.Textbox(label="Page", value="No generations yet.", interactive=False)
            history_table = gr.Dataframe(
                headers=HISTORY_HEADERS,
                value=[],
                type="array",
                interactive=False,
                wrap=True,
                label="Recent generations",
            )
            history_detail_box = gr.Textbox(label="History detail", lines=14, interactive=False)
            history_audio = gr.Audio(label="History audio", type="filepath", interactive=False)
            delete_message = gr.Textbox(label="Delete", interactive=False)
            with gr.Row():
                delete_button = gr.Button("Delete history record")
                delete_confirm = gr.Button("Confirm delete", variant="stop")

        with gr.Column(visible=False, elem_classes=["vocalis-page"]) as monitor_page:
            gr.Markdown("## System Monitor\nReadings refresh every 5 seconds while this page is open.")
            monitor_box = gr.Textbox(label="Runtime", lines=14, interactive=False)

        with gr.Column(visible=False, elem_classes=["vocalis-page"]) as settings_page:
            gr.Markdown("## Settings")
            settings_box = gr.Textbox(
                label="Application settings",
                value=settings_text(services),
                lines=16,
                interactive=False,
            )

        with gr.Column(visible=True) as job_panel:
            gr.Markdown("## Current job")
            job_status = gr.Textbox(label="Status", value="Idle", interactive=False)
            job_detail = gr.Textbox(label="Job detail", lines=12, interactive=False, value=_job_idle())
            job_audio = gr.Audio(label="Output audio", type="filepath", interactive=False)
            cancel_button = gr.Button("Cancel pending job", interactive=False)

        job_timer = gr.Timer(JOB_REFRESH_SECONDS, active=False)
        monitor_timer = gr.Timer(MONITOR_REFRESH_SECONDS, active=False)

        pages = [design_page, clone_page, history_page_col, monitor_page, settings_page]
        buttons = [nav_design, nav_clone, nav_history, nav_monitor, nav_settings]
        nav_outputs = [
            *pages,
            *buttons,
            job_panel,
            monitor_timer,
            brand,
            history_table,
            history_ids,
            history_summary,
            history_offset,
            selected_history,
            armed_delete,
            history_detail_box,
            history_audio,
            delete_message,
            monitor_box,
            settings_box,
        ]

        def _open(page: str, offset: int, type_label: str, status_label: str, provider_label: str):
            return _navigation(
                services,
                page,
                offset=0 if page == "history" else offset,
                type_label=type_label,
                status_label=status_label,
                provider_label=provider_label,
            )

        nav_inputs = [history_offset, type_filter, status_filter, provider_filter]
        nav_design.click(lambda offset, kind, status, provider: _open("design", offset, kind, status, provider), nav_inputs, nav_outputs)
        nav_clone.click(lambda offset, kind, status, provider: _open("clone", offset, kind, status, provider), nav_inputs, nav_outputs)
        nav_history.click(lambda offset, kind, status, provider: _open("history", offset, kind, status, provider), nav_inputs, nav_outputs)
        nav_monitor.click(lambda offset, kind, status, provider: _open("monitor", offset, kind, status, provider), nav_inputs, nav_outputs)
        nav_settings.click(lambda offset, kind, status, provider: _open("settings", offset, kind, status, provider), nav_inputs, nav_outputs)

        design_generate.click(
            lambda text, description, cfg, steps, normalize, denoise, current: _submit_view(
                submit_design(services, text, description, cfg, steps, normalize, denoise),
                services,
                current,
            ),
            [design_text, design_description, design_cfg, design_steps, design_normalize, design_denoise, job_id],
            [job_id, job_status, job_detail, job_audio, cancel_button, job_timer],
        )
        design_reset.click(
            lambda: ("", "", 2, 10, False, False),
            outputs=[design_text, design_description, design_cfg, design_steps, design_normalize, design_denoise],
        )
        clone_generate.click(
            lambda text, reference, transcript, cfg, steps, normalize, denoise, current: _submit_view(
                submit_clone(services, text, reference, transcript, cfg, steps, normalize, denoise),
                services,
                current,
            ),
            [clone_text, clone_file, clone_transcript, clone_cfg, clone_steps, clone_normalize, clone_denoise, job_id],
            [job_id, job_status, job_detail, job_audio, cancel_button, job_timer],
        )
        clone_reset.click(
            lambda: ("", None, "", None, "Upload a WAV, FLAC, or MP3 reference.", 2, 10, False, False),
            outputs=[
                clone_text,
                clone_file,
                clone_transcript,
                reference_audio,
                reference_meta,
                clone_cfg,
                clone_steps,
                clone_normalize,
                clone_denoise,
            ],
        )
        clone_file.change(
            lambda path: inspect_reference(services, path),
            [clone_file],
            [reference_meta, reference_audio],
        )
        job_timer.tick(
            lambda current: _job_tick(inspect_job(services, current)),
            [job_id],
            [job_status, job_detail, job_audio, cancel_button, job_timer],
        )
        cancel_button.click(
            lambda current: _job_update(cancel_job(services, current)),
            [job_id],
            [job_id, job_status, job_detail, job_audio, cancel_button, job_timer],
        )
        monitor_timer.tick(lambda: monitor_text(services), outputs=monitor_box)

        def _reload_history(offset: int, kind: str, status: str, provider: str):
            page = history_page(
                services,
                offset=offset,
                type_label=kind,
                status_label=status,
                provider_label=provider,
            )
            return _history_update(page, "", "", None, "")

        history_inputs = [history_offset, type_filter, status_filter, provider_filter]
        history_outputs = [
            history_table,
            history_ids,
            history_summary,
            history_offset,
            selected_history,
            armed_delete,
            history_detail_box,
            history_audio,
            delete_message,
        ]
        for control in (type_filter, status_filter, provider_filter):
            control.change(
                lambda kind, status, provider: _reload_history(0, kind, status, provider),
                [type_filter, status_filter, provider_filter],
                history_outputs,
            )
        history_prev.click(
            lambda offset, kind, status, provider: _reload_history(
                offset - HISTORY_PAGE_SIZE, kind, status, provider
            ),
            history_inputs,
            history_outputs,
        )
        history_next.click(
            lambda offset, kind, status, provider: _reload_history(
                offset + HISTORY_PAGE_SIZE, kind, status, provider
            ),
            history_inputs,
            history_outputs,
        )

        def _on_select(event: gr.SelectData, ids: list[str]):
            return _select_history(services, event, ids)

        history_table.select(
            _on_select,
            [history_ids],
            [selected_history, armed_delete, history_detail_box, history_audio, delete_message],
        )
        def _on_delete(selected, armed, offset, kind, status, provider):
            return _delete(services, selected, armed, offset, kind, status, provider)

        delete_inputs = [selected_history, armed_delete, history_offset, type_filter, status_filter, provider_filter]
        delete_outputs = [
            selected_history,
            armed_delete,
            delete_message,
            history_table,
            history_ids,
            history_summary,
            history_offset,
            history_detail_box,
            history_audio,
        ]
        delete_button.click(_on_delete, delete_inputs, delete_outputs)
        delete_confirm.click(_on_delete, delete_inputs, delete_outputs)

    return demo


def launch_app(settings: Settings, services: StudioServices) -> None:
    """Start the local Gradio server. Public sharing stays off."""
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    create_app(settings, services).launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        share=False,
        inbrowser=False,
        show_error=True,
        theme=studio_theme(),
        css=css,
        footer_links=[],
        max_file_size=settings.max_upload_bytes,
        allowed_paths=[str(settings.output_dir), str(settings.upload_dir)],
    )


def _parameters():
    with gr.Accordion("Inference parameters", open=False):
        cfg = gr.Slider(minimum=0.1, maximum=5, value=2, step=0.1, label="CFG value")
        steps = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="Inference timesteps")
        normalize = gr.Checkbox(value=False, label="Normalize")
        denoise = gr.Checkbox(value=False, label="Denoise")
        gr.Markdown(_PARAMETER_NOTE)
    return cfg, steps, normalize, denoise


def _page_intro(title: str, settings: Settings, mock_copy: str) -> str:
    if settings.model_provider is ModelProvider.MOCK:
        return f"## {title}\n\n{mock_copy}"
    return f"## {title}"


def _job_idle() -> str:
    return "No job yet. Generate to queue one. Refresh does not submit a new job."


def _submit_view(view: JobView, services: StudioServices, current_id: str):
    """Keep the current job on screen when a new submission is rejected."""
    if view.job_id:
        return _job_update(view)
    current = inspect_job(services, current_id)
    detail = view.detail if not current.job_id else f"{view.detail}\n\n{current.detail}"
    return (
        current.job_id,
        view.status_label if not current.job_id else current.status_label,
        detail,
        current.audio_path,
        gr.update(interactive=current.can_cancel),
        gr.update(active=current.refresh),
    )


def _job_update(view: JobView):
    return (
        view.job_id,
        view.status_label,
        view.detail,
        view.audio_path,
        gr.update(interactive=view.can_cancel),
        gr.update(active=view.refresh),
    )


def _job_tick(view: JobView):
    return (
        view.status_label,
        view.detail,
        view.audio_path,
        gr.update(interactive=view.can_cancel),
        gr.update(active=view.refresh),
    )


def _navigation(
    services: StudioServices,
    page: str,
    *,
    offset: int,
    type_label: str,
    status_label: str,
    provider_label: str,
):
    visible = [gr.update(visible=shown) for shown in page_visibility(page)]
    variants = [gr.update(variant=variant) for variant in button_variants(page)]
    history = (
        history_page(
            services,
            offset=offset,
            type_label=type_label,
            status_label=status_label,
            provider_label=provider_label,
        )
        if page == "history"
        else None
    )
    return (
        *visible,
        *variants,
        gr.update(visible=page in {"design", "clone"}),
        gr.update(active=page == "monitor"),
        render_shell(services.settings, services.manager),
        gr.skip() if history is None else history.rows,
        gr.skip() if history is None else list(history.ids),
        gr.skip() if history is None else history.summary,
        gr.skip() if history is None else history.offset,
        "" if page == "history" else gr.skip(),
        "" if page == "history" else gr.skip(),
        gr.skip() if history is None else "Select a row to read the full record.",
        gr.skip() if history is None else None,
        gr.skip() if history is None else "",
        monitor_text(services) if page == "monitor" else gr.skip(),
        settings_text(services) if page == "settings" else gr.skip(),
    )


def _history_update(page: HistoryPage, selected: str, armed: str, audio: str | None, message: str):
    return (
        page.rows,
        list(page.ids),
        page.summary,
        page.offset,
        selected,
        armed,
        "Select a row to read the full record." if not selected else "",
        audio,
        message,
    )


def _select_history(services: StudioServices, event: gr.SelectData, ids: list[str]):
    index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    if not isinstance(index, int) or index < 0 or index >= len(ids):
        return "", "", "Select a row to read the full record.", None, ""
    job_id = ids[index]
    detail, audio = history_detail(services, job_id)
    return job_id, "", detail, audio, ""


def _delete(
    services: StudioServices,
    selected: str,
    armed: str,
    offset: int,
    type_label: str,
    status_label: str,
    provider_label: str,
):
    outcome: DeleteOutcome = delete_history(services, selected, armed)
    if not outcome.deleted:
        detail, audio = history_detail(services, outcome.selected_id) if outcome.selected_id else ("", None)
        if not outcome.selected_id:
            detail = "Select a row to read the full record."
        return (
            outcome.selected_id,
            outcome.armed_id,
            outcome.message,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            detail,
            audio,
        )
    page = history_page(
        services,
        offset=offset,
        type_label=type_label,
        status_label=status_label,
        provider_label=provider_label,
    )
    return (
        "",
        "",
        outcome.message,
        page.rows,
        list(page.ids),
        page.summary,
        page.offset,
        "Select a row to read the full record.",
        None,
    )
