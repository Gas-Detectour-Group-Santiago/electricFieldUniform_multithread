#!/usr/bin/env python3
"""PySide6 interface for electric-field campaigns and GIF generation."""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys

from PySide6.QtCore import QProcess, QSettings, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "campaign.yaml"
PYTHON = sys.executable

RUNTIME_OPTION_DEFAULTS = {
    "space_charge": False,
    "record_excitation_positions": True,
    "measure_gas_transport": False,
}


def read_yaml_bool(text: str, key: str, default: bool) -> bool:
    match = re.search(
        rf"(?m)^[ \t]*{re.escape(key)}[ \t]*:[ \t]*(true|false|yes|no|on|off|1|0)[ \t]*(?:#.*)?$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return default
    return match.group(1).lower() in {"true", "yes", "on", "1"}


def write_yaml_bool(text: str, key: str, value: bool) -> str:
    """Update one top-level boolean while preserving any inline comment."""
    rendered = "true" if value else "false"
    pattern = re.compile(
        rf"(?m)^(?P<indent>[ \t]*){re.escape(key)}[ \t]*:[ \t]*"
        rf"(?:true|false|yes|no|on|off|1|0)(?P<tail>[ \t]*(?:#.*)?)$",
        flags=re.IGNORECASE,
    )
    if pattern.search(text):
        return pattern.sub(
            lambda match: f"{match.group('indent')}{key}: {rendered}{match.group('tail')}",
            text,
            count=1,
        )

    line = f"{key}: {rendered}\n"
    scan_mode = re.search(r"(?m)^scan_mode[ \t]*:.*(?:\n|$)", text)
    if scan_mode:
        position = scan_mode.end()
        return text[:position] + line + text[position:]
    return line + text


class CampaignTab(QWidget):
    MIN_FONT_POINTS = 8
    MAX_FONT_POINTS = 20

    def __init__(self):
        super().__init__()
        self.settings = QSettings("MaximumVoltage", "electricUniform")

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished)

        self.config_path = DEFAULT_CONFIG
        initial_yaml = DEFAULT_CONFIG.read_text(encoding="utf-8")
        self.editor = QPlainTextEdit()
        self.editor.setPlainText(initial_yaml)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self.space_charge = QCheckBox("Space charge")
        self.space_charge.setToolTip(
            "Include charged rings during every primary avalanche. "
            "This changes the gain physics and uses a separate alpha fit."
        )
        self.excitation_positions = QCheckBox("Excitation positions")
        self.excitation_positions.setToolTip(
            "Write hExcXY and hExcZT. hLevels and the total excitation count "
            "are kept even when this is disabled."
        )
        self.gas_transport = QCheckBox("Magboltz transport")
        self.gas_transport.setToolTip(
            "Calculate drift velocity, diffusion, Townsend and attachment "
            "at the exact simulated field. Disabled by default."
        )
        self.load_campaign_options(initial_yaml)

        self.open_button = QPushButton("Open YAML")
        self.save_button = QPushButton("Save YAML")
        self.zoom_out_button = QPushButton("A−")
        self.zoom_reset_button = QPushButton("100 %")
        self.zoom_in_button = QPushButton("A+")
        self.toggle_yaml_button = QPushButton("Hide YAML")
        self.run_button = QPushButton("Run campaign")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        self.open_button.clicked.connect(self.open_yaml)
        self.save_button.clicked.connect(self.save_yaml)
        self.zoom_out_button.clicked.connect(lambda: self.change_font_size(-1))
        self.zoom_reset_button.clicked.connect(self.reset_font_size)
        self.zoom_in_button.clicked.connect(lambda: self.change_font_size(+1))
        self.toggle_yaml_button.clicked.connect(self.toggle_yaml)
        self.run_button.clicked.connect(self.run)
        self.stop_button.clicked.connect(self.stop)

        self.shortcuts = [
            QShortcut(QKeySequence("Ctrl++"), self),
            QShortcut(QKeySequence("Ctrl+="), self),
            QShortcut(QKeySequence("Ctrl+-"), self),
            QShortcut(QKeySequence("Ctrl+0"), self),
        ]
        self.shortcuts[0].activated.connect(lambda: self.change_font_size(+1))
        self.shortcuts[1].activated.connect(lambda: self.change_font_size(+1))
        self.shortcuts[2].activated.connect(lambda: self.change_font_size(-1))
        self.shortcuts[3].activated.connect(self.reset_font_size)

        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.save_button)
        buttons.addSpacing(12)
        buttons.addWidget(self.zoom_out_button)
        buttons.addWidget(self.zoom_reset_button)
        buttons.addWidget(self.zoom_in_button)
        buttons.addWidget(self.toggle_yaml_button)
        buttons.addStretch()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)

        options = QHBoxLayout()
        options.addWidget(QLabel("Campaign:"))
        options.addWidget(self.space_charge)
        options.addWidget(self.excitation_positions)
        options.addWidget(self.gas_transport)
        options.addStretch()

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "Mixture", "Composition", "p [bar]", "gap [mm]", "Target",
            "E [kV/cm]", "Gain", "npe", "Progress", "Status", "Details",
        ])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(42)
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(70)
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(10, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 105)
        self.table.setColumnWidth(1, 245)
        self.table.setColumnWidth(2, 85)
        self.table.setColumnWidth(3, 90)
        self.table.setColumnWidth(4, 105)
        self.table.setColumnWidth(5, 115)
        self.table.setColumnWidth(6, 105)
        self.table.setColumnWidth(7, 75)
        self.table.setColumnWidth(8, 280)
        self.table.setColumnWidth(9, 135)

        simulations_panel = QWidget()
        simulations_layout = QVBoxLayout(simulations_panel)
        simulations_layout.setContentsMargins(0, 0, 0, 0)
        simulations_title = QLabel("Simulations")
        font = simulations_title.font()
        font.setBold(True)
        simulations_title.setFont(font)
        simulations_layout.addWidget(simulations_title)
        simulations_layout.addWidget(self.table)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.editor)
        self.splitter.addWidget(simulations_panel)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)
        saved_sizes = self.settings.value("campaign/splitter_sizes")
        if isinstance(saved_sizes, list) and len(saved_sizes) == 2:
            self.splitter.setSizes([int(value) for value in saved_sizes])
        else:
            self.splitter.setSizes([330, 500])
        self.splitter.splitterMoved.connect(
            lambda *_: self.settings.setValue(
                "campaign/splitter_sizes", self.splitter.sizes()
            )
        )

        self.rows = {}
        self.job_rows = {}
        self.job_progress: dict[int, QProgressBar] = {}
        self.job_maximum: dict[int, int] = {}
        self.output_buffer = ""
        self.process_log: list[str] = []
        self.scan_mode = "gain"
        self.total_targets = 0
        self.accepted_targets: set[tuple] = set()
        self.running_jobs: set[int] = set()
        self.saved_attempts = 0

        self.progress = QProgressBar()
        self.progress.setMinimumHeight(30)
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.status = QLabel("Ready")
        self.status.setMinimumHeight(24)

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addLayout(options)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self.font_points = int(self.settings.value("campaign/font_points", 10))
        self.apply_font_size()

    def open_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open campaign", str(self.config_path.parent), "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        self.config_path = Path(path)
        text = self.config_path.read_text(encoding="utf-8")
        self.load_campaign_options(text)
        self.editor.setPlainText(text)

    def load_campaign_options(self, text: str | None = None):
        if text is None:
            text = self.editor.toPlainText()
        self.space_charge.setChecked(read_yaml_bool(text, "space_charge", False))
        self.excitation_positions.setChecked(
            read_yaml_bool(text, "record_excitation_positions", True)
        )
        self.gas_transport.setChecked(
            read_yaml_bool(text, "measure_gas_transport", False)
        )

    def save_yaml(self):
        text = self.editor.toPlainText()
        text = write_yaml_bool(text, "space_charge", self.space_charge.isChecked())
        text = write_yaml_bool(
            text, "record_excitation_positions", self.excitation_positions.isChecked()
        )
        text = write_yaml_bool(
            text, "measure_gas_transport", self.gas_transport.isChecked()
        )
        if not text.endswith("\n"):
            text += "\n"
        self.editor.setPlainText(text)
        self.config_path.write_text(text, encoding="utf-8")
        self.status.setText(f"Saved: {self.config_path}")

    def change_font_size(self, delta: int):
        self.font_points = max(
            self.MIN_FONT_POINTS,
            min(self.MAX_FONT_POINTS, self.font_points + delta),
        )
        self.apply_font_size()

    def reset_font_size(self):
        self.font_points = 10
        self.apply_font_size()

    def apply_font_size(self):
        for widget in (self.editor, self.table, self.table.horizontalHeader(), self.status):
            font = widget.font()
            font.setPointSize(self.font_points)
            widget.setFont(font)
        self.zoom_reset_button.setText(f"{self.font_points * 10} %")
        self.table.verticalHeader().setDefaultSectionSize(
            max(38, int(4.1 * self.font_points))
        )
        for bar in self.job_progress.values():
            font = bar.font()
            font.setPointSize(self.font_points)
            bar.setFont(font)
            bar.setMinimumHeight(max(24, int(2.8 * self.font_points)))
        self.settings.setValue("campaign/font_points", self.font_points)

    def toggle_yaml(self):
        visible = not self.editor.isVisible()
        self.editor.setVisible(visible)
        self.toggle_yaml_button.setText("Hide YAML" if visible else "Show YAML")
        if visible:
            self.splitter.setSizes([330, 500])

    def set_running_controls(self, running: bool):
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.open_button.setEnabled(not running)
        self.space_charge.setEnabled(not running)
        self.excitation_positions.setEnabled(not running)
        self.gas_transport.setEnabled(not running)

    def run(self):
        self.save_yaml()
        self.rows.clear()
        self.job_rows.clear()
        self.job_progress.clear()
        self.job_maximum.clear()
        self.table.setRowCount(0)
        self.output_buffer = ""
        self.process_log.clear()
        self.total_targets = 0
        self.accepted_targets.clear()
        self.running_jobs.clear()
        self.saved_attempts = 0
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Configuring and building uniformE…")
        self.set_running_controls(True)
        self.status.setText("Configuring and building uniformE…")

        arguments = [
            "run_campaign.py",
            str(self.config_path),
            "--space-charge" if self.space_charge.isChecked() else "--no-space-charge",
            "--excitation-positions" if self.excitation_positions.isChecked()
            else "--no-excitation-positions",
            "--gas-transport" if self.gas_transport.isChecked()
            else "--no-gas-transport",
        ]
        self.process.start(PYTHON, arguments)

    def stop(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            self.status.setText("Stopping active workers…")

    def read_output(self):
        self.output_buffer += bytes(
            self.process.readAllStandardOutput()
        ).decode("utf-8", errors="replace")
        lines = self.output_buffer.split("\n")
        self.output_buffer = lines.pop()
        for line in lines:
            if line.startswith("CAMPAIGN_EVENT "):
                try:
                    self.handle_event(json.loads(line[len("CAMPAIGN_EVENT "):]))
                except json.JSONDecodeError:
                    self.process_log.append(line)
            elif line.strip():
                self.process_log.append(line)
        self.process_log = self.process_log[-200:]

    def read_error(self):
        text = bytes(self.process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if text.strip():
            lines = text.strip().splitlines()
            self.process_log.extend(lines)
            self.process_log = self.process_log[-200:]
            self.status.setText(lines[-1])

    def progress_bar(self, row: int, job_id: int) -> QProgressBar:
        bar = self.job_progress.get(job_id)
        if bar is None:
            bar = QProgressBar()
            bar.setTextVisible(True)
            bar.setMinimumHeight(max(24, int(2.8 * self.font_points)))
            font = bar.font()
            font.setPointSize(self.font_points)
            bar.setFont(font)
            self.job_progress[job_id] = bar
            self.table.setCellWidget(row, 8, bar)
        return bar

    def update_campaign_progress(self):
        if self.total_targets <= 0:
            self.progress.setRange(0, 0)
            return
        accepted = min(len(self.accepted_targets), self.total_targets)
        self.progress.setRange(0, self.total_targets)
        self.progress.setValue(accepted)
        self.progress.setFormat(
            f"Campaign: {accepted}/{self.total_targets} targets accepted · "
            f"{len(self.running_jobs)} running · {self.saved_attempts} ROOT attempts saved · %p%"
        )

    def target_key(self, event: dict) -> tuple:
        mode = str(event.get("scan_mode", self.scan_mode))
        target = (
            event.get("target_field_kv_cm")
            if mode == "field" else event.get("target_gain")
        )
        return (
            event.get("mixture"), event.get("composition"),
            event.get("pressure_bar"), event.get("gap_mm"), mode, target,
        )

    def handle_event(self, event):
        event_type = event.get("type")
        if event_type == "build_started":
            self.status.setText("Configuring and building uniformE…")
            return
        if event_type == "build_finished":
            self.status.setText("Build completed · starting campaign")
            return
        if event_type == "build_failed":
            message = event.get("error", "CMake/build failed")
            self.status.setText(message.splitlines()[-1])
            QMessageBox.critical(self, "Build failed", message)
            return
        if event_type == "campaign_started":
            self.scan_mode = str(event.get("scan_mode", "gain"))
            self.total_targets = int(event.get("targets", 0))
            self.accepted_targets.clear()
            self.running_jobs.clear()
            self.saved_attempts = 0
            target_title = (
                "Target E [kV/cm]" if self.scan_mode == "field" else "Target gain"
            )
            self.table.setHorizontalHeaderItem(4, QTableWidgetItem(target_title))
            self.update_campaign_progress()
            self.status.setText(
                f"{self.scan_mode.capitalize()} mode · "
                f"{event['families']} families · {self.total_targets} targets · "
                f"{event['workers']} workers"
            )
            return
        if event_type == "campaign_finished":
            completed = bool(event.get("completed"))
            self.status.setText(
                "Completed" if completed
                else f"Finished with {event.get('remaining_targets', 0)} pending targets"
            )
            if completed and self.total_targets > 0:
                self.progress.setRange(0, self.total_targets)
                self.progress.setValue(self.total_targets)
                self.progress.setFormat(
                    f"Campaign completed · {self.total_targets}/{self.total_targets} targets · "
                    f"{self.saved_attempts} new ROOT attempts saved · 100%"
                )
            else:
                self.update_campaign_progress()
            return
        if event_type == "prediction":
            self.status.setText(
                f"Predictor: {event.get('predictor', '')} · "
                f"E/p = {float(event.get('reduced_field_kv_cm_bar', 0.0)):.3g} "
                "kV cm⁻¹ bar⁻¹"
            )
            return

        job_id = event.get("job_id")
        if event_type in {"transport_started", "transport_finished", "progress"}:
            row = self.job_rows.get(job_id)
            if row is None:
                return
            bar = self.progress_bar(row, int(job_id))
            if event_type == "transport_started":
                bar.setRange(0, 0)
                bar.setFormat("Magboltz transport…")
                self.table.setItem(row, 9, QTableWidgetItem("Magboltz"))
                self.status.setText(
                    f"Job {job_id}: Magboltz transport at "
                    f"{float(event.get('field_v_cm', 0.0)) / 1000.0:.3f} kV/cm"
                )
                return
            if event_type == "transport_finished":
                maximum = max(1, self.job_maximum.get(int(job_id), 1))
                bar.setRange(0, maximum)
                bar.setValue(0)
                seconds = float(event.get("seconds", 0.0))
                bar.setFormat(f"Magboltz done in {seconds:.1f} s · avalanche pending")
                self.table.setItem(row, 9, QTableWidgetItem("Starting avalanche"))
                self.status.setText(f"Job {job_id}: Magboltz completed in {seconds:.1f} s")
                return

            current = int(event.get("current", 0))
            maximum = max(1, int(event.get("maximum", 1)))
            bar.setRange(0, maximum)
            bar.setValue(min(current, maximum))
            running_gain = event.get("running_gain")
            relative_error = event.get("relative_error")
            phase = "Field run" if self.scan_mode == "field" else "Refining"
            bar.setFormat(f"{phase} · %v/%m npe · %p%")
            if isinstance(running_gain, (int, float)) and running_gain == running_gain:
                self.table.setItem(row, 6, QTableWidgetItem(f"{float(running_gain):.5g}"))
            self.table.setItem(row, 7, QTableWidgetItem(str(current)))
            self.table.setItem(row, 9, QTableWidgetItem("Running"))
            error_text = ""
            if (
                isinstance(relative_error, (int, float))
                and relative_error == relative_error
                and relative_error != float("inf")
            ):
                error_text = f" · σG/G={100.0 * float(relative_error):.2f}%"
            self.status.setText(
                f"Job {job_id}: {current}/{maximum} npe"
                + (f" · G={float(running_gain):.5g}" if isinstance(running_gain, (int, float)) and running_gain == running_gain else "")
                + error_text
            )
            return

        if event_type not in {"started", "result", "failed"}:
            return

        key = self.target_key(event)
        event_mode = str(event.get("scan_mode", self.scan_mode))
        target_value = key[-1]
        # Keep one visible row per simulation attempt. A refinement of the
        # same target therefore appears as a new row instead of overwriting
        # the previous measured (E, G) point.
        row = self.job_rows.get(job_id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setRowHeight(row, max(38, int(4.1 * self.font_points)))
        if job_id is not None:
            self.job_rows[job_id] = row

        status = "Running"
        details = ""
        if event_type == "started":
            self.running_jobs.add(int(job_id))
            maximum = max(1, int(event.get("max_npe", 1)))
            self.job_maximum[int(job_id)] = maximum
        elif event_type == "failed":
            self.running_jobs.discard(int(job_id))
            attempt = int(event.get("attempt", 1))
            status = f"Retrying ({attempt}/3)" if event.get("will_retry", False) else "Failed"
            error = event.get("error", "Unknown error")
            details = error.strip().splitlines()[-1]
        elif event_type == "result":
            self.running_jobs.discard(int(job_id))
            self.saved_attempts += 1
            accepted = bool(event.get("accepted"))
            if accepted:
                self.accepted_targets.add(key)
            status = "Done" if event_mode == "field" else (
                "Accepted" if accepted else "Saved · Refining"
            )
            details = event.get("root", "")

        field = event.get("field_v_cm")
        field_kv_cm = "" if field is None else float(field) / 1000.0
        values = [
            event.get("mixture", ""),
            event.get("composition", ""),
            event.get("pressure_bar", ""),
            event.get("gap_mm", ""),
            "" if target_value is None else target_value,
            "" if field_kv_cm == "" else f"{field_kv_cm:.4g}",
            event.get("gain", ""),
            event.get("npe", ""),
            "",
            status,
            details,
        ]
        for column, value in enumerate(values):
            if column == 8:
                continue
            self.table.setItem(row, column, QTableWidgetItem(str(value)))

        if job_id is not None:
            bar = self.progress_bar(row, int(job_id))
            if event_type == "started":
                maximum = self.job_maximum[int(job_id)]
                bar.setRange(0, maximum)
                bar.setValue(0)
                bar.setFormat("Queued · 0/%m npe")
            elif event_type == "result":
                bar.setRange(0, 100)
                bar.setValue(100)
                bar.setFormat("ROOT saved · accepted" if event.get("accepted") else "ROOT saved · refining")
            elif event_type == "failed":
                bar.setRange(0, 100)
                bar.setValue(0)
                bar.setFormat(status)

        if event_type == "failed":
            error = event.get("error", "Unknown error")
            for column in (9, 10):
                item = self.table.item(row, column)
                if item is not None:
                    item.setToolTip(error)
            self.status.setText(details)

        self.update_campaign_progress()

    def finished(self, exit_code, *_):
        self.set_running_controls(False)
        self.update_campaign_progress()
        if exit_code != 0:
            message = "\n".join(self.process_log[-40:]).strip()
            if not message:
                message = self.status.text() or f"Campaign exited with code {exit_code}"
            QMessageBox.critical(self, "Campaign failed", message)


class GifTab(QWidget):
    GAS_OPTIONS = [
        ("Argon", "ar"),
        ("Helium", "he"),
        ("CF4", "cf4"),
        ("N2", "n2"),
        ("CO2", "co2"),
        ("CH4", "ch4"),
        ("Isobutane", "ic4h10"),
        ("C2H2F4", "c2h2f4"),
    ]

    def __init__(self):
        super().__init__()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished)

        self.gas1 = self.gas_box(allow_empty=False)
        self.gas2 = self.gas_box(allow_empty=True)
        self.gas3 = self.gas_box(allow_empty=True)
        self.set_gas(self.gas1, "ar")
        self.set_gas(self.gas2, "cf4")
        self.set_gas(self.gas3, "")

        self.composition1 = self.double_box(0.0, 100.0, 99.0, 4)
        self.composition2 = self.double_box(0.0, 100.0, 1.0, 4)
        self.composition3 = self.double_box(0.0, 100.0, 0.0, 4)

        self.pressure = self.double_box(0.001, 20.0, 1.0, 3)
        self.gap = self.double_box(0.001, 10.0, 0.05, 3)
        self.height = self.double_box(1.0, 10.0, 1.5, 2)
        self.npe = QSpinBox()
        self.npe.setRange(1, 10000)
        self.npe.setValue(1)
        self.tmax = self.double_box(0.001, 1.0e6, 10.0, 3)
        self.frames = QSpinBox()
        self.frames.setRange(2, 500)
        self.frames.setValue(80)
        self.space_charge = QCheckBox()
        self.move_ions = QCheckBox()
        self.move_ions.setChecked(True)
        self.ion_speed = self.double_box(0.0, 1.0, 1.0e-4, 8)
        self.ion_speed.setSingleStep(1.0e-5)
        self.ion_speed.setSuffix(" cm/ns")
        self.move_ions.toggled.connect(self.ion_speed.setEnabled)

        self.mode = QComboBox()
        self.mode.addItems(["Electric field", "Target gain"])
        self.value = self.double_box(0.001, 5000.0, 40.0, 4)
        self.mode.currentIndexChanged.connect(self.update_value_label)
        self.value_label = QLabel("Field [kV/cm]")
        self.total_label = QLabel("Total composition: 100 %")

        for box in (self.composition1, self.composition2, self.composition3):
            box.valueChanged.connect(self.update_total)
        for box in (self.gas2, self.gas3):
            box.currentIndexChanged.connect(self.update_total)

        form = QFormLayout()
        form.addRow("Gas 1", self.gas1)
        form.addRow("Gas 1 fraction [%]", self.composition1)
        form.addRow("Gas 2", self.gas2)
        form.addRow("Gas 2 fraction [%]", self.composition2)
        form.addRow("Gas 3", self.gas3)
        form.addRow("Gas 3 fraction [%]", self.composition3)
        form.addRow("", self.total_label)
        form.addRow("Pressure [bar]", self.pressure)
        form.addRow("Gap [mm]", self.gap)
        form.addRow("Height factor", self.height)
        form.addRow("Primary electrons", self.npe)
        form.addRow("Field or gain", self.mode)
        form.addRow(self.value_label, self.value)
        form.addRow("t max [ns]", self.tmax)
        form.addRow("Frames", self.frames)
        form.addRow("Move ions", self.move_ions)
        form.addRow("Constant ion speed", self.ion_speed)
        form.addRow("Space charge", self.space_charge)

        self.generate_button = QPushButton("Generate GIF")
        self.generate_button.clicked.connect(self.generate)
        self.status = QLabel("Ready")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.status)
        layout.addStretch()
        self.update_total()

    @classmethod
    def gas_box(cls, allow_empty: bool) -> QComboBox:
        box = QComboBox()
        if allow_empty:
            box.addItem("None", "")
        for label, identifier in cls.GAS_OPTIONS:
            box.addItem(label, identifier)
        return box

    @staticmethod
    def set_gas(box: QComboBox, identifier: str) -> None:
        index = box.findData(identifier)
        if index >= 0:
            box.setCurrentIndex(index)

    @staticmethod
    def double_box(minimum, maximum, value, decimals):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setValue(value)
        return box

    def update_value_label(self):
        self.value_label.setText(
            "Field [kV/cm]" if self.mode.currentIndex() == 0 else "Target gain"
        )

    def composition(self) -> list[tuple[str, float]]:
        pairs = [
            (str(self.gas1.currentData()), self.composition1.value()),
            (str(self.gas2.currentData()), self.composition2.value()),
            (str(self.gas3.currentData()), self.composition3.value()),
        ]
        active = [(gas, value) for gas, value in pairs if gas]
        gases = [gas for gas, _ in active]
        if len(gases) != len(set(gases)):
            raise ValueError("The same gas cannot be selected twice")
        if any(value < 0.0 for _, value in active):
            raise ValueError("Gas fractions cannot be negative")
        total = sum(value for _, value in active)
        if abs(total - 100.0) > 1.0e-6:
            raise ValueError(f"Gas fractions must add to 100 %, obtained {total:g} %")
        if not any(value > 0.0 for _, value in active):
            raise ValueError("At least one gas fraction must be positive")
        return active

    def update_total(self):
        values = [self.composition1.value()]
        if self.gas2.currentData():
            values.append(self.composition2.value())
        if self.gas3.currentData():
            values.append(self.composition3.value())
        total = sum(values)
        self.total_label.setText(f"Total composition: {total:g} %")

    def generate(self):
        try:
            components = self.composition()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid composition", str(error))
            return

        component_argument = ",".join(
            f"{gas}:{fraction:.12g}" for gas, fraction in components
        )
        arguments = [
            "run_campaign.py", "--gif",
            "--components", component_argument,
            "--pressure-bar", str(self.pressure.value()),
            "--gap-mm", str(self.gap.value()),
            "--npe", str(self.npe.value()),
            "--height-factor", str(self.height.value()),
            "--tmax-ns", str(self.tmax.value()),
            "--frames", str(self.frames.value()),
        ]
        if self.mode.currentIndex() == 0:
            arguments += ["--field-kv-cm", str(self.value.value())]
        else:
            arguments += ["--gain", str(self.value.value())]
        arguments += ["--ion-speed-cm-ns", str(self.ion_speed.value())]
        arguments.append("--move-ions" if self.move_ions.isChecked() else "--no-move-ions")
        arguments.append("--space-charge" if self.space_charge.isChecked() else "--no-space-charge")

        self.generate_button.setEnabled(False)
        self.status.setText("Generating GIF")
        self.process.start(PYTHON, arguments)

    def read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            self.status.setText(lines[-1])

    def read_error(self):
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text.strip():
            self.status.setText(text.strip().splitlines()[-1])

    def finished(self, exit_code, *_):
        self.generate_button.setEnabled(True)
        if exit_code != 0:
            QMessageBox.critical(self, "GIF", self.status.text())


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("electricUniform")
        self.resize(1050, 760)

        tabs = QTabWidget()
        tabs.addTab(CampaignTab(), "Campaign")
        tabs.addTab(GifTab(), "GIF")
        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
