"""Create an explicitly synthetic end-to-end study for desktop integration QA.

Run with the analysis Python environment. This creates new local fixture
data only and never reads or changes a real study.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

ANALYSIS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ANALYSIS))
from wearable_pipeline import CSV_COLUMNS,SessionWriter,prepare_dataset
from mlcookbook_export import export_mlcookbook
from movement_evaluate import evaluate_export


def create(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    sessions=[]
    for subject in range(6):
        writer=SessionWriter(output/"synthetic-sessions",{
            "subject":f"synthetic-p{subject:02}","sample_rate_hz":50,"placement":"belt_fixture",
            "orientation":"synthetic axes; no physical board","enclosure":"synthetic",
            "firmware":"synthetic","synthetic":True,"purpose":"software integration only"})
        for class_id,label in enumerate(("standing","walking","jumping")):
            for sample in range(150):
                seconds=sample/50
                wave=math.sin(seconds*(class_id+1)*2*math.pi+subject*0.1)
                values=dict.fromkeys(CSV_COLUMNS,"")
                values.update(ts_ms=class_id*5000+sample*20,label=label,
                    ax=wave*(class_id+0.1),ay=wave*0.3,az=9.80665+wave*class_id*2,
                    gx=wave*(class_id+1)*3,gy=wave*2,gz=wave*class_id)
                writer.feed((",".join(str(values[c]) for c in CSV_COLUMNS)+"\n").encode())
        writer.close()
        sessions.append(writer.path)
    profile=json.loads((ANALYSIS/"profiles/belt-v1.json").read_text())
    prepare_dataset(sessions,output/"prepared",channels=6,window=100,stride=25,profile=profile)
    exported=export_mlcookbook(output/"prepared",output/"synthetic-movement-study")
    assert exported["synthetic"] is True
    evaluated=evaluate_export(output/"synthetic-movement-study",output/"evaluation")
    (output/"README.md").write_text(
        "# SYNTHETIC integration fixture\n\nOpen synthetic-movement-study/export.json in the Movement Study app. "
        "These invented signals demonstrate the software workflow only. Scores do not measure real activity "
        "or fall recognition. Do not merge these sessions into real training data.\n",encoding="utf-8")
    print(json.dumps({"synthetic":True,"manifest":str((output/"synthetic-movement-study/export.json").resolve()),
        "native_evaluator":evaluated["native_evaluator"],"test_holdout_sealed":not evaluated["final_test_evaluated"],
        "features":len(exported["features"])},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    create(args.out)


if __name__=="__main__":main()
