import json
import tempfile
import unittest
import zlib
from pathlib import Path

from flash_transfer import DumpReader, import_flash
from serial_control import command, stop_for_inspection
from test_wearable import row
from wearable_pipeline import load_session, SessionWriter, prepare_dataset


def frame(sequence,session,kind,text,dropped=0):
    payload=f"{sequence},{session},{dropped},{kind}|{text}"
    return f"[flash] v2 crc32={zlib.crc32(payload.encode()):08x} payload={payload}\n"


def dump(records,generation=1,gaps=0):
    return (f"[flash] dump begin version=2 generation={generation} used={len(records)} valid={len(records)-gaps} damaged={gaps}\n"
            +"".join(records)+f"[flash] dump complete version=2 generation={generation} used={len(records)}\n")


class FlashTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self,text):
        path=self.root/"dump.log"
        path.write_text(text,encoding="utf-8")
        return path

    def test_complete_session_and_post_stop_alert_keep_raw_lineage(self):
        records=[frame(1,0,4,"[flash] boot ts_ms=0 label="),
                 frame(2,2,2,"[flash] session ts_ms=100 label=walking"),
                 frame(3,2,1,row(120).strip()),frame(4,2,3,"[flash] stop ts_ms=140 label="),
                 frame(5,2,5,"[fall] state=TEST ts_ms=160 source_ts_ms=160 confidence=1.000")]
        batch,result=import_flash(self.source(dump(records)),self.root/"out",{"subject":"p1","sample_rate_hz":50})
        self.assertEqual(result["sessions"][0]["status"],"imported")
        path=batch/result["sessions"][0]["path"]
        metadata,rows=load_session(path)
        self.assertEqual(rows[0]["ts_ms"],"120")
        self.assertEqual(metadata["counts"]["fall_events"],1)
        self.assertIn(b"crc32=",(path/"raw.log").read_bytes())
        self.assertTrue(metadata["flash_wire_crc_verified"])

    def test_torn_session_is_failed_but_next_boot_session_recovers(self):
        records=[frame(1,1,2,"[flash] session ts_ms=100 label=walking"),frame(2,1,1,row(120).strip()),
                 "[flash] gap sequence=3\n",frame(4,0,4,"[flash] boot ts_ms=0 label="),
                 frame(5,5,2,"[flash] session ts_ms=100 label=walking"),frame(6,5,1,row(120).strip()),
                 frame(7,5,3,"[flash] stop ts_ms=140 label=")]
        batch,result=import_flash(self.source(dump(records,gaps=1)),self.root/"out",{})
        self.assertEqual([s["status"] for s in result["sessions"]],["failed","imported"])
        self.assertIn("damaged",result["sessions"][0]["error"])
        self.assertEqual(load_session(batch/result["sessions"][1]["path"])[0]["counts"]["rows"],1)

    def test_loss_counter_quarantines_session(self):
        records=[frame(1,1,2,"[flash] session ts_ms=100 label=walking",4),
                 frame(2,1,1,row(120).strip(),5),frame(3,1,3,"[flash] stop ts_ms=140 label=",5)]
        _,result=import_flash(self.source(dump(records)),self.root/"out",{})
        self.assertEqual(result["sessions"][0]["status"],"failed")
        self.assertIn("dropped",result["sessions"][0]["error"])

    def test_crc_truncation_sequence_and_extent_rejected_before_output(self):
        original=dump([frame(1,1,2,"[flash] session ts_ms=100 label=walking"),frame(2,1,1,row(120).strip()),
                       frame(3,1,3,"[flash] stop ts_ms=140 label=")])
        bad=(original.replace("walking","jumping",1),original.split("[flash] dump complete")[0],
             original.replace("used=3 valid=3","used=4 valid=4"),
             original.replace(frame(2,1,1,row(120).strip()),frame(4,1,1,row(120).strip())))
        for text in bad:
            with self.assertRaises(ValueError):
                import_flash(self.source(text),self.root/"out",{})
            self.assertFalse((self.root/"out").exists())

    def test_serial_ack_handles_fragments_and_explicit_failure(self):
        class Connection:
            in_waiting=1
            def __init__(self,chunks):self.chunks=iter(chunks);self.writes=[]
            def write(self,data):self.writes.append(data)
            def read(self,_):return next(self.chunks)
        received=bytearray()
        connection=Connection([b"stop\r\n[cli] logging sto",b"pped committed=1\r\n"])
        command(connection,"stop","[cli] logging stopped",received.extend)
        self.assertEqual(connection.writes,[b"stop\r\n"])
        self.assertIn(b"committed=1",received)
        with self.assertRaisesRegex(RuntimeError,"not committed"):
            command(Connection([b"[cli] ERROR stop not committed\n"]),"stop","stopped",received.extend)
        recovery=Connection([b"[cli] ERROR stop not committed; data requires recovery\n",
                             b"[cli] mode=csv enabled=0 sink=flash\n"])
        self.assertFalse(stop_for_inspection(recovery,received.extend))
        self.assertEqual(recovery.writes,[b"stop\r\n",b"status\r\n"])
        with self.assertRaisesRegex(RuntimeError,"requires recording stopped"):
            stop_for_inspection(Connection([b"[cli] ERROR stop not committed\n",
                                            b"[cli] mode=csv enabled=1 sink=flash\n"]),received.extend)

    def test_valid_wire_crc_does_not_hide_bad_payloads_or_capture_times(self):
        for i,(sample,stop) in enumerate((("not a sample",140),(row(120,acceleration=float("nan")).strip(),140),
                                         (row(80).strip(),140),(row(120).strip(),110))):
            records=[frame(1,1,2,"[flash] session ts_ms=100 label=walking"),frame(2,1,1,sample),
                     frame(3,1,3,f"[flash] stop ts_ms={stop} label=")]
            _,result=import_flash(self.source(dump(records)),self.root/f"case-{i}",{})
            self.assertEqual(result["sessions"][0]["status"],"failed")
            self.assertTrue(result["sessions"][0]["error"])

    def test_failed_sessions_require_explicit_exclusion_and_stay_in_provenance(self):
        paths=[]
        for i in range(4):
            writer=SessionWriter(self.root/"sessions",{"subject":f"p{i}","sample_rate_hz":50})
            writer.feed("".join(row(i*10000+j*20) for j in range(125)).encode())
            writer.close("complete" if i<3 else "failed","interrupted" if i==3 else None)
            paths.append(writer.path)
        with self.assertRaisesRegex(ValueError,"finalized"):
            prepare_dataset(paths,self.root/"bad")
        result=prepare_dataset(paths,self.root/"good",exclude_failed=True)
        self.assertEqual(result["excluded_sessions"][0]["status"],"failed")
        self.assertEqual(result["excluded_sessions"][0]["error"],"interrupted")


if __name__=="__main__":unittest.main()
