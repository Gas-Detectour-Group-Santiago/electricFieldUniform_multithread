#!/usr/bin/env python3
"""Small PySide6 interface for campaigns and GIF generation."""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
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
    "measure_gas_transport": True,
}


def read_yaml_bool(text: str, key: str, default: bool) -> bool:
    """Read a legacy top-level YAML boolean."""
    match = re.search(
        rf"(?m)^[ \t]*{re.escape(key)}[ \t]*:[ \t]*(true|false|yes|no|on|off|1|0)[ \t]*(?:#.*)?$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return default
    return match.group(1).lower() in {"true", "yes", "on", "1"}


def strip_runtime_options(text: str) -> str:
    """Keep runtime switches in the GUI, not duplicated in the YAML editor."""
    for key in RUNTIME_OPTION_DEFAULTS:
        text = re.sub(
            rf"(?m)^[ \t]*{re.escape(key)}[ \t]*:.*(?:\n|$)",
            "",
            text,
        )
    return text.rstrip() + "\n"


class CampaignTab(QWidget):
    def __init__(self):
        super().__init__()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished)

        self.config_path = DEFAULT_CONFIG
        initial_yaml = DEFAULT_CONFIG.read_text(encoding="utf-8")
        self.editor = QPlainTextEdit()

        self.space_charge = QCheckBox("Space charge")
        self.space_charge.setToolTip(
            "Include charged rings during every primary avalanche. "
            "This changes the gain physics and uses a separate alpha fit."
        )
        self.excitation_positions = QCheckBox("Excitation positions")
        self.excitation_positions.setChecked(True)
        self.excitation_positions.setToolTip(
            "Write hExcXY and hExcZT. hLevels and the total excitation count "
            "are kept even when this is disabled."
        )
        self.gas_transport = QCheckBox("Magboltz transport")
        self.gas_transport.setChecked(True)
        self.gas_transport.setToolTip(
            "Query electron drift velocity, longitudinal/transverse diffusion, "
            "Townsend and attachment coefficients once per simulated point."
        )
        self.load_campaign_options(initial_yaml)
        self.editor.setPlainText(strip_runtime_options(initial_yaml))

        self.open_button = QPushButton("Open YAML")
        self.save_button = QPushButton("Save YAML")
        self.run_button = QPushButton("Run campaign")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        self.open_button.clicked.connect(self.open_yaml)
        self.save_button.clicked.connect(self.save_yaml)
        self.run_button.clicked.connect(self.run)
        self.stop_button.clicked.connect(self.stop)

        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.save_button)
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
            "Mixture", "Fraction", "p [bar]", "gap [mm]", "Target",
            "E [kV/cm]", "Gain", "npe", "Progress", "Status", "Details",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.rows = {}
        self.job_rows = {}
        self.job_progress: dict[int, QProgressBar] = {}
        self.output_buffer = ""
        self.process_log: list[str] = []
        self.scan_mode = "gain"

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.status = QLabel("Ready")

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addLayout(options)
        layout.addWidget(self.editor, 2)
        layout.addWidget(self.table, 3)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

    def open_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open campaign", str(self.config_path.parent), "YAML (*.yaml *.yml)"
        )
        if not path:
            return
        self.config_path = Path(path)
        text = self.config_path.read_text(encoding="utf-8")
        self.load_campaign_options(text)
        self.editor.setPlainText(strip_runtime_options(text))

    def load_campaign_options(self, text: str | None = None):
        if text is None:
            text = self.editor.toPlainText()
        self.space_charge.setChecked(
            read_yaml_bool(text, "space_charge", False)
        )
        self.excitation_positions.setChecked(
            read_yaml_bool(text, "record_excitation_positions", True)
        )
        self.gas_transport.setChecked(
            read_yaml_bool(text, "measure_gas_transport", True)
        )

    def save_yaml(self):
        # The YAML describes only the campaign grid. Runtime/output switches
        # belong to the checkboxes above and are passed as command-line options.
        text = strip_runtime_options(self.editor.toPlainText())
        self.editor.setPlainText(text)
        self.config_path.write_text(text, encoding="utf-8")
        self.status.setText(f"Saved: {self.config_path}")

    def run(self):
        self.save_yaml()
        self.rows.clear()
        self.job_rows.clear()
        self.job_progress.clear()
        self.table.setRowCount(0)
        self.output_buffer = ""
        self.process_log.clear()
        self.progress.setVisible(True)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
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

    def read_output(self):
        self.output_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
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
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text.strip():
            lines = text.strip().splitlines()
            self.process_log.extend(lines)
            self.process_log = self.process_log[-200:]
            self.status.setText(lines[-1])

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
            target_title = (
                "Target E [kV/cm]" if self.scan_mode == "field"
                else "Target gain"
            )
            self.table.setHorizontalHeaderItem(4, QTableWidgetItem(target_title))
            self.status.setText(
                f"{self.scan_mode.capitalize()} mode · "
                f"{event['families']} families · {event['targets']} targets · "
                f"{event['workers']} workers"
            )
            return
        if event_type == "campaign_finished":
            self.status.setText(
                "Completed" if event.get("completed")
                else f"Finished with {event.get('remaining_targets', 0)} pending targets"
            )
            return

        if event_type == "prediction":
            self.status.setText(
                f"Predictor: {event.get('predictor', '')} · "
                f"E/p = {float(event.get('reduced_field_kv_cm_bar', 0.0)):.3g} "
                "kV cm⁻¹ bar⁻¹"
            )
            return

        if event_type == "progress":
            job_id = event.get("job_id")
            row = self.job_rows.get(job_id)
            if row is not None:
                current = int(event.get("current", 0))
                maximum = max(1, int(event.get("maximum", 1)))
                bar = self.job_progress.get(job_id)
                if bar is None:
                    bar = QProgressBar()
                    bar.setTextVisible(True)
                    self.job_progress[job_id] = bar
                    self.table.setCellWidget(row, 8, bar)
                bar.setRange(0, maximum)
                bar.setValue(min(current, maximum))
                bar.setFormat("%v/%m npe  (%p%)")
                self.table.setItem(row, 9, QTableWidgetItem("Running"))
            return

        if event_type not in {"started", "result", "failed"}:
            return

        event_mode = str(event.get("scan_mode", self.scan_mode))
        target_value = (
            event.get("target_field_kv_cm")
            if event_mode == "field" else event.get("target_gain")
        )
        key = (
            event.get("mixture"), event.get("fraction"), event.get("pressure_bar"),
            event.get("gap_mm"), event_mode, target_value,
        )
        job_id = event.get("job_id")
        row = self.job_rows.get(job_id)

        if row is None and key in self.rows:
            row = self.rows[key]

        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.rows[key] = row

        if job_id is not None:
            self.job_rows[job_id] = row

        status = "Running"
        details = ""
        if event_type == "failed":
            attempt = int(event.get("attempt", 1))
            if event.get("will_retry", False):
                status = f"Retrying ({attempt}/3)"
            else:
                status = "Failed"
            error = event.get("error", "Unknown error")
            details = error.strip().splitlines()[-1]

            # A failed attempt does not carry gain/npe and may not carry the
            # proposed field. Never erase the values already shown by started.
            identity_values = [
                event.get("mixture", ""),
                event.get("fraction", ""),
                event.get("pressure_bar", ""),
                event.get("gap_mm", ""),
                "" if target_value is None else target_value,
            ]
            for column, value in enumerate(identity_values):
                if self.table.item(row, column) is None:
                    self.table.setItem(row, column, QTableWidgetItem(str(value)))
            bar = self.job_progress.get(job_id)
            if bar is not None:
                if event.get("will_retry", False):
                    bar.setFormat("Retrying")
                else:
                    bar.setFormat("Failed")
            self.table.setItem(row, 9, QTableWidgetItem(status))
            self.table.setItem(row, 10, QTableWidgetItem(details))
            for column in (9, 10):
                self.table.item(row, column).setToolTip(error)
            self.status.setText(details)
            return

        if event_type == "result":
            if event_mode == "field":
                status = "Done"
            else:
                status = "Accepted" if event.get("accepted") else "Refining"
            details = event.get("root", "")

        field = event.get("field_v_cm")
        field_kv_cm = "" if field is None else float(field) / 1000.0
        values = [
            event.get("mixture", ""),
            event.get("fraction", ""),
            event.get("pressure_bar", ""),
            event.get("gap_mm", ""),
            "" if target_value is None else target_value,
            field_kv_cm,
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
            bar = self.job_progress.get(job_id)
            if bar is None:
                bar = QProgressBar()
                bar.setTextVisible(True)
                self.job_progress[job_id] = bar
                self.table.setCellWidget(row, 8, bar)
            if event_type == "started":
                maximum = max(1, int(event.get("max_npe", 1)))
                bar.setRange(0, maximum)
                bar.setValue(0)
                bar.setFormat("0/%m npe  (0%)")
            elif event_type == "result":
                bar.setRange(0, 100)
                bar.setValue(100)
                bar.setFormat("Completed")

    def finished(self, exit_code, *_):
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if exit_code != 0:
            message = "\n".join(self.process_log[-40:]).strip()
            if not message:
                message = self.status.text() or f"Campaign exited with code {exit_code}"
            QMessageBox.critical(self, "Campaign failed", message)


class GifTab(QWidget):
    def __init__(self):
        super().__init__()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished)

        self.mixture = QComboBox()
        self.mixture.addItems([
            "ArCF4", "ArN2", "HeCF4", "ArCO2", "ArCH4", "ArIso",
            "ArC2H2F4",
        ])

        self.fraction = self.double_box(0.0, 100.0, 1.0, 3)
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
        self.ion_speed.setEnabled(True)
        self.move_ions.toggled.connect(self.ion_speed.setEnabled)

        self.mode = QComboBox()
        self.mode.addItems(["Electric field", "Target gain"])
        self.value = self.double_box(0.001, 5000.0, 40.0, 4)
        self.mode.currentIndexChanged.connect(self.update_value_label)
        self.value_label = QLabel("Field [kV/cm]")

        form = QFormLayout()
        form.addRow("Mixture", self.mixture)
        form.addRow("Additive fraction [%]", self.fraction)
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

    def generate(self):
        arguments = [
            "run_campaign.py", "--gif",
            "--mixture", self.mixture.currentText(),
            "--fraction", str(self.fraction.value()),
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
        if self.space_charge.isChecked():
            arguments.append("--space-charge")

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
