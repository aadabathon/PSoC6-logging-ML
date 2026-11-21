import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parents[2]/"workingDir"/"tools"
if not all((TOOLS / name).is_file() for name in ("prepare_model.py", "demo.py")):
    raise unittest.SkipTest("requires the sibling firmware checkout's model preparation tools")
sys.path.insert(0,str(TOOLS))
from prepare_model import prepare_model
from demo import check_model

COMPILER=shutil.which("gcc") or (r"C:\msys64\ucrt64\bin\gcc.exe" if Path(r"C:\msys64\ucrt64\bin\gcc.exe").is_file() else None)


@unittest.skipUnless(COMPILER,"host C compiler unavailable")
class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.export=self.root/"export"
        self.export.mkdir()
        (self.export/"model.h").write_text(
            "#define IMAI_API_QUEUE\n#define IMAI_DATA_IN_COUNT 6\n#define IMAI_DATA_OUT_COUNT 2\n"
            "#define IMAI_RET_SUCCESS 0\n#define IMAI_RET_NODATA -1\n"
            "int IMAI_init(void);int IMAI_enqueue(const float *);int IMAI_dequeue(float *);\n")
        (self.export/"model.c").write_text(
            '#include "model.h"\nstatic int ready;static float value;'
            'int IMAI_init(void){ready=0;return 0;}int IMAI_enqueue(const float*x){value=x[0];ready=1;return 0;}'
            'int IMAI_dequeue(float*y){if(!ready)return -1;ready=0;y[0]=1-value;y[1]=value;return 0;}')
        self.inputs=self.root/"input.data"
        self.inputs.write_text("Time (seconds),CH0,CH1,CH2,CH3,CH4,CH5\n0,0,0,0,0,0,0\n0.02,1,0,0,0,0,0\n")
        self.expected=self.root/"output.data"
        self.expected.write_text("Time (seconds),Duration (seconds),pred_standing,pred_falling\n0,1,1,0\n0.02,1,0,1\n")
        self.profile=self.root/"profile.json"
        self.profile.write_text(json.dumps({"channels":6,"sample_rate_hz":50,"units":["g"]*3+["deg/s"]*3,
            "features":["ax","ay","az","gx","gy","gz"],"window_samples":100,"stride_samples":25,
            "allowed_labels":["standing","falling"]}))

    def tearDown(self):self.temp.cleanup()

    def test_replays_actual_c_and_installs_only_after_parity(self):
        firmware=self.root/"firmware"
        (firmware/"tools").mkdir(parents=True)
        shutil.copyfile(TOOLS/"replay_model.c",firmware/"tools"/"replay_model.c")
        with patch("prepare_model.ROOT",firmware):
            result=prepare_model(self.export,self.inputs,self.expected,self.profile,compiler=COMPILER,install=True)
        self.assertTrue(result["installed"])
        self.assertEqual(result["classes"],["standing","falling"])
        self.assertEqual(result["max_absolute_error"],0)
        self.assertIn('"standing", "falling"',(firmware/"source/ml/generated/contract.h").read_text())
        self.assertTrue((firmware/"source/ml/generated/model-ready.json").is_file())
        profiles=self.root/"analysis/profiles"
        profiles.mkdir(parents=True)
        shutil.copyfile(self.profile,profiles/"belt-v1.json")
        self.assertEqual(check_model(firmware)["classes"],["standing","falling"])
        with (firmware/"source/ml/generated/model.c").open("a") as stream:
            stream.write("\n/* changed after parity */\n")
        with self.assertRaisesRegex(ValueError,"changed since"):
            check_model(firmware)

    def test_bad_expected_outputs_or_profile_do_not_install(self):
        self.expected.write_text("pred_standing,pred_falling\n0.5,0.5\n0.5,0.5\n")
        with self.assertRaisesRegex(ValueError,"exceeds tolerance"):
            prepare_model(self.export,self.inputs,self.expected,self.profile,compiler=COMPILER)
        profile=json.loads(self.profile.read_text());profile["units"]=["m/s^2"]*6
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError,"physical units"):
            prepare_model(self.export,self.inputs,self.expected,self.profile,compiler=COMPILER)


if __name__=="__main__":unittest.main()
