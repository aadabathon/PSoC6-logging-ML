import unittest
from hardware_check import assess


class AcceptanceTests(unittest.TestCase):
    def rows(self,times):return [{"ts_ms":str(t)} for t in times]

    def test_regular_stream_passes_but_gaps_rejects_and_short_capture_fail(self):
        self.assertTrue(assess(self.rows(range(0,10000,20)),{},10)["passed"])
        values=list(range(0,10000,20));values[250]+=20
        self.assertFalse(assess(self.rows(values),{},10)["passed"])
        self.assertFalse(assess(self.rows(range(0,10000,20)),{"rejected":1},10)["passed"])
        self.assertFalse(assess(self.rows(range(0,1000,20)),{},10)["passed"])


if __name__=="__main__":unittest.main()
