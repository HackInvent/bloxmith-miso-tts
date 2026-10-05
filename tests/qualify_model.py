"""Explicit real-model smoke test. Never downloads assets or contacts a speech API.

Default is a read-only prerequisite report. --execute needs authorized assets,
the exact dedicated environment and a private watermark key through stdin/TTY.
This does not replace framework suites or listening to the generated speech.
"""

import argparse
import getpass
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types

package=types.ModuleType("miso_qualification_owned")
package.__path__=[str(Path(__file__).resolve().parents[1])];sys.modules[package.__name__]=package
from miso_qualification_owned import assets_store,logic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--python",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    parser.add_argument("--results-directory",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--key-stdin",action="store_true",help="Read the private five-byte JSON key from stdin instead of a hidden TTY prompt.")
    args=parser.parse_args()
    try:
        inventory=assets_store.inventory(args.directory)
        inventory["inference_python_exists"]=args.python.is_file()
        if not args.execute:
            print(json.dumps(inventory,indent=2));return 0 if inventory["ready"] and inventory["inference_python_exists"] else 1
        if not inventory["ready"]:raise logic.MisoError("assets_missing")
        if not inventory["inference_python_exists"]:raise logic.MisoError("dependency")
        if args.results_directory is None:parser.error("--execute requires --results-directory")
        key=sys.stdin.readline(130).strip() if args.key_stdin else getpass.getpass("Private watermark JSON (hidden): ")
        logic.watermark_key(lambda _:key,"qualification")
        results=args.results_directory.absolute();results.mkdir(parents=True,exist_ok=True)
        cfg=logic.configuration({"asset_directory":str(args.directory.absolute()),"environment_python":str(args.python.absolute()),
            "device":args.device,"watermark_key_ref":"qualification","output_directory":str(results/"audio"),"timeout_sec":1800})
        with TemporaryDirectory(prefix="miso-qualification-",dir=results) as temporary:
            prepared=logic.generate({"text":"This is a local speech synthesis test. The report is ready.","request_id":"real-model-smoke"},
                cfg,results,Path(temporary),lambda _:key)
            try:metadata=logic.publish(prepared)
            finally:prepared.close()
        probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-of","json",metadata["path"]]))
        stream=probe["streams"][0]
        if stream["codec_name"]!="opus" or int(stream["sample_rate"])!=48000 or stream["channels"]!=1:
            raise logic.MisoError("audio_invalid")
        report={"status":"passed","scope":"actual pinned model smoke inference and complete Opus file, not quality/watermark certification",
            "metadata":metadata,"manual_review_required":"Listen to the complete speech; record hardware, dependency provenance, Stop and runtime integration separately."}
        print(json.dumps(report,indent=2));return 0
    except (logic.MisoError,assets_store.AssetError,OSError,ValueError,subprocess.SubprocessError) as exc:
        code=exc.code if isinstance(exc,logic.MisoError) else str(exc) if isinstance(exc,assets_store.AssetError) else "qualification_failed"
        print(json.dumps({"status":"failed","code":code}));return 1


if __name__=="__main__":raise SystemExit(main())
