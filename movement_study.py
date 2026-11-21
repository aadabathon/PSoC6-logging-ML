"""Standalone desktop app for verified, fixed-group wearable studies."""

from __future__ import annotations

import argparse
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, ttk

FOLD_CHOICES = ("5", "3", "2", "10", "0")


def read_export(path):
    from movement_baselines import load_wearable_study
    from movement_snapshot import Study

    path = Path(path).expanduser().resolve()
    if (path.is_dir() and (path / "study.json").is_file()) or path.name == "study.json":
        from offline_study import verify_study

        path = Path(verify_study(path)["movement_study"])
    frame, imported = load_wearable_study(path)
    return Study(path.name if path.is_dir() else path.parent.name, frame, imported.to_dict())


def _report_text(report):
    selected = report["selected_model"]
    validation = report["candidates"][selected]["validation"]
    lines = [f"Selected {selected.replace('_', ' ')} by macro F1 on fixed validation groups.", "",
             "Fixed validation groups",
             f"  Macro F1 {validation['macro_f1']:.4f}"
             f"    Balanced accuracy {validation['balanced_accuracy']:.4f}"
             f"    Accuracy {validation['accuracy']:.4f}", "",
             f"{'Candidate':24} {'Macro F1':>8} {'Balanced acc.':>14} {'Group folds':>12}"]
    for name, score in report["candidates"].items():
        metrics = score["validation"]
        lines.append(f"{name.replace('_', ' '):24} {metrics['macro_f1']:8.4f} "
                     f"{metrics['balanced_accuracy']:14.4f} {len(score['grouped_cv']):12}")
    if report["final_test_evaluated"]:
        frozen = report["candidates"][selected]["test"]
        lines.extend(("", "Frozen final test groups",
                      f"  Macro F1 {frozen['macro_f1']:.4f}"
                      f"    Balanced accuracy {frozen['balanced_accuracy']:.4f}"
                      f"    Accuracy {frozen['accuracy']:.4f}"))
    else:
        lines.extend(("", "Final test sealed for this run."))

    labels = report["labels"]
    width = max(8, max(map(len, labels)) + 2)
    lines.extend(("", "Validation confusion matrix (actual rows, predicted columns)",
                  " " * width + "".join(f"{label:>{width}}" for label in labels)))
    for label, row in zip(labels, validation["confusion_matrix"]):
        lines.append(f"{label:<{width}}" + "".join(f"{value:>{width}}" for value in row))
    return lines


class MovementWindow:
    def __init__(self, root):
        self.root = root
        self.study = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="movement")
        self.pending = None
        self.closing = False
        self.folds = tk.StringVar(root, "5")
        self.seed = tk.StringVar(root, "42")
        self.final = tk.BooleanVar(root, False)
        self.input_note = tk.StringVar(root, "Open an export.json, an offline study, or a saved snapshot.")
        self.status = tk.StringVar(root)
        self.result_note = tk.StringVar(root)
        root.title("Movement Study")
        root.minsize(700, 460)
        root.geometry("920x690")
        root.protocol("WM_DELETE_WINDOW", self.close)

        body = ttk.Frame(root, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Movement Study", font=("TkDefaultFont", 18)).pack(anchor="w")
        actions = ttk.Frame(body)
        actions.pack(anchor="w", pady=(10, 8))
        self.open_export = self._button(actions, "Open study…", self.choose_study)
        self.open_saved = self._button(actions, "Open snapshot…", self.choose_snapshot)
        self.save = self._button(actions, "Save snapshot…", self.save_snapshot)
        self.export = self._button(actions, "Export evaluation…", self.export_result)
        ttk.Label(body, textvariable=self.input_note, wraplength=670).pack(fill="x", pady=(0, 10))

        controls = ttk.LabelFrame(body, text="Grouped evaluation", padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="Training group folds").grid(row=0, column=0, sticky="w")
        self.fold_control = ttk.Combobox(controls, textvariable=self.folds, values=FOLD_CHOICES, width=5)
        self.fold_control.grid(row=0, column=1, padx=(7, 18))
        ttk.Label(controls, text="Random seed").grid(row=0, column=2, sticky="w")
        self.seed_control = ttk.Entry(controls, textvariable=self.seed, width=13)
        self.seed_control.grid(row=0, column=3, padx=(7, 18))
        self.final_control = ttk.Checkbutton(controls, text="Evaluate reserved test groups", variable=self.final)
        self.final_control.grid(row=0, column=4, sticky="w")
        ttk.Label(controls, text="0 disables training CV. Reserve test evaluation for frozen model decisions.",
                  wraplength=640).grid(row=1, column=0, columnspan=5, sticky="w", pady=(7, 0))
        self.run = ttk.Button(body, text="Run grouped baselines", command=self.run_baselines)
        self.run.pack(anchor="w", pady=8)
        ttk.Label(body, textvariable=self.result_note).pack(anchor="w")

        results = ttk.Frame(body)
        results.pack(fill="both", expand=True, pady=(4, 6))
        results.rowconfigure(0, weight=1)
        results.columnconfigure(0, weight=1)
        self.report = tk.Text(results, wrap="none", font="TkFixedFont", padx=8, pady=8, state="disabled")
        vertical = ttk.Scrollbar(results, command=self.report.yview)
        horizontal = ttk.Scrollbar(results, orient="horizontal", command=self.report.xview)
        self.report.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.report.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        ttk.Label(body, textvariable=self.status, wraplength=670).pack(fill="x")
        for variable in (self.folds, self.seed, self.final):
            variable.trace_add("write", self.refresh_actions)
        self.refresh_actions()

    @staticmethod
    def _button(parent, label, command):
        button = ttk.Button(parent, text=label, command=command)
        button.pack(side="left", padx=(0, 7))
        return button

    def settings(self):
        from movement_snapshot import Settings

        try:
            return Settings(int(self.folds.get()), self.final.get(), int(self.seed.get()))
        except (ValueError, tk.TclError):
            raise ValueError("Choose a fold count and enter a random seed from 0 to 2147483647.") from None

    def refresh_actions(self, *_):
        busy = self.pending is not None
        try:
            settings = self.settings() if self.study is not None else None
        except ValueError:
            settings = None
        has_result = self.study is not None and self.study.report is not None
        for widget, enabled in (
            (self.open_export, not busy), (self.open_saved, not busy),
            (self.save, not busy and settings is not None), (self.export, not busy and has_result),
            (self.run, not busy and settings is not None),
            (self.seed_control, not busy and self.study is not None),
            (self.final_control, not busy and self.study is not None),
        ):
            widget.configure(state="normal" if enabled else "disabled")
        self.fold_control.configure(state="readonly" if not busy and self.study is not None else "disabled")
        if self.study is None:
            self.result_note.set("")
        elif settings is None:
            self.result_note.set("Enter a valid random seed to run or save.")
        elif not has_result:
            self.result_note.set("No evaluation yet. The final test is sealed until explicitly evaluated.")
        elif settings != self.study.run_settings:
            self.result_note.set("Settings changed. Results below are from the previous run.")
        else:
            self.result_note.set("")

    def submit(self, message, operation, accept):
        if self.pending is not None:
            return
        self.status.set(message)
        self.pending = self.worker.submit(operation)
        self.refresh_actions()
        self.root.after(30, self.finish, accept)

    def finish(self, accept):
        if not self.pending.done():
            self.root.after(30, self.finish, accept)
            return
        try:
            value = self.pending.result()
            if not self.closing:
                accept(value)
        except Exception as error:
            self.status.set(str(error))
        finally:
            self.pending = None
            self.refresh_actions()
            if self.closing:
                self.close()

    def close(self):
        if self.pending is not None:
            self.closing = True
            self.status.set("Closing after the current operation finishes…")
        else:
            self.worker.shutdown(wait=False)
            self.root.destroy()

    def choose_study(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open export.json or study.json",
                                          filetypes=[("Study manifests", "*.json")])
        if path:
            self.load_export(path)

    def choose_snapshot(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open Movement Study snapshot",
                                          filetypes=[("Movement Study snapshots", "*.mlcs *.mlcb")])
        if path:
            self.load_snapshot(path)

    def load_export(self, path):
        self.submit("Verifying study files and fixed groups…", lambda: read_export(path), self.show_study)

    def load_snapshot(self, path):
        from movement_snapshot import open_study

        self.submit("Checking the saved dataset snapshot…", lambda: open_study(path), self.show_study)

    def show_study(self, study):
        self.study = study
        self.folds.set(str(study.settings.cv_folds))
        self.final.set(study.settings.final_test)
        self.seed.set(str(study.settings.random_state))
        group = study.imported["group_by"]
        self.input_note.set(f"{study.name} · {len(study.frame):,} windows · "
                            f"{len(study.imported['features'])} features · grouped by {group}")
        self.draw_report()
        self.status.set("Study ready.")

    def draw_report(self):
        from movement_baselines import META

        current = self.study
        lines = []
        if current.frame[META["synthetic"]].iloc[0]:
            lines.extend(("Synthetic fixture. Scores only check the software workflow.", ""))
        if current.report is not None:
            lines.extend(_report_text(current.report))
            lines.extend(("", "Notes", *current.report["warnings"], *current.report["limitations"], ""))
        if current.version_notes:
            lines.extend((*current.version_notes, ""))
        lines.append("Fixed group assignments")
        for group, split in sorted(current.imported["group_splits"].items()):
            lines.append(f"  {group}    {split}")
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", "\n".join(lines))
        self.report.configure(state="disabled")

    def run_baselines(self):
        from movement_baselines import evaluate_wearable

        data = self.study
        settings = self.settings()

        def run():
            return evaluate_wearable(data.frame, **asdict(settings)).to_dict()

        def accept(report):
            self.study = replace(data, settings=settings, report=report, run_settings=settings, version_notes=())
            self.draw_report()
            self.status.set("Grouped evaluation complete.")

        self.submit("Fitting and evaluating grouped baselines…", run, accept)

    def save_snapshot(self):
        from movement_snapshot import save_study

        path = filedialog.asksaveasfilename(parent=self.root, title="Save Movement Study snapshot",
                                            defaultextension=".mlcs", filetypes=[("Movement Study", "*.mlcs")])
        if path:
            saved = replace(self.study, settings=self.settings())
            self.submit("Saving snapshot…", lambda: save_study(path, saved),
                        lambda _: self.status.set(f"Saved {Path(path).name}."))

    def export_result(self):
        from movement_snapshot import export_evaluation

        path = filedialog.asksaveasfilename(parent=self.root, title="Export evaluation",
                                            initialfile="movement-evaluation.json", defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if path:
            report = self.study.report
            self.submit("Exporting evaluation…", lambda: export_evaluation(path, report),
                        lambda _: self.status.set(f"Exported {Path(path).name}."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--study", type=Path, help="export.json, study.json, or their folder")
    source.add_argument("--experiment", type=Path, help="saved .mlcs or legacy Movement Study .mlcb")
    options = parser.parse_args(argv)

    root = tk.Tk()
    app = MovementWindow(root)
    if options.study:
        root.after_idle(app.load_export, options.study)
    if options.experiment:
        root.after_idle(app.load_snapshot, options.experiment)
    root.mainloop()


if __name__ == "__main__":
    main()
