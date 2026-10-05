"""FB1/FB2/FB3/FB4: adapter contracts and genuine processes/codecs; model is simulated."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"tests"),str(Path(__file__).parent)]
from blocs.miso_tts import logic,worker
from miso_fixture import SECRET,settings,alive,wait,started,check_file
from test_assets import AssetTests
from test_listener import ListenerTests


def refused(action,code):
    try:action()
    except logic.MisoError as exc:
        assert exc.code==code,(exc.code,code);return
    raise AssertionError("Invalid operation accepted: "+code)


class Tensor:
    """Small NumPy-backed test double for the worker's final-audio tensor interface."""
    def __init__(self,value):self.value=np.asarray(value)
    @property
    def ndim(self):return self.value.ndim
    def numel(self):return self.value.size
    def detach(self):return self
    def to(self,*args,**kwargs):
        dtype=kwargs.get("dtype",args[0] if args else None)
        return Tensor(self.value.astype(dtype)) if dtype else self
    def abs(self):return Tensor(np.abs(self.value))
    def max(self):return self.value.max()
    def __mul__(self,value):return Tensor(self.value*value)
    def round(self):return Tensor(self.value.round())
    def numpy(self):return self.value


def adapter_tests(root):
    torch=NS(all=np.all,isfinite=lambda tensor:np.isfinite(tensor.value),float32=np.float32,int16=np.int16,bfloat16="bf16")
    cfg=logic.configuration({})
    class Generator:
        sample_rate=24000
        def __init__(self,audio,eos=True):
            frames=iter([np.ones((1,32)),np.zeros((1,32))]);self._model=NS(generate_frame=lambda:next(frames))
            self.audio=audio;self.eos=eos
        def generate(self,**kw):
            assert kw==dict(text="Example",speaker=0,context=[],max_audio_length_ms=30000,temperature=.9,topk=50)
            self._model.generate_frame()
            if self.eos:self._model.generate_frame()
            return Tensor(self.audio)
    good=root/"adapter";good.mkdir()
    info=worker.generate_audio(Generator(np.sin(np.arange(12000)/12)*1.2),torch,"Example",cfg,good)
    assert info["gain"]<1 and info["frames"]==12000
    logic.audio_info(good/"generated.wav",cfg,json.dumps(info))
    refused(lambda:worker.generate_audio(Generator([.1],False),torch,"Example",cfg,root),"audio_limit")
    for audio in ([],[0.],[[.2]],[float("nan")],[float("inf")],np.ones(24000*31)):
        refused(lambda:worker.generate_audio(Generator(audio),torch,"Example",cfg,root),"audio_invalid")
    model=NS(generate_frame=lambda:np.zeros((1,31)))
    worker.guard_eos(model,torch);refused(model.generate_frame,"model_failed")
    model=NS(generate_frame=lambda:np.zeros((1,32)))
    state=worker.guard_eos(model,torch);model.generate_frame();assert state["ended"]
    refused(model.generate_frame,"model_failed")
    # Exercise exact local loader arguments; third-party SDKs are explicitly doubled.
    calls={}
    generator=NS(load_miso_8b=lambda **kw:calls.update(load=kw) or "loaded")
    def tokenizer(path,**kw):calls["tokenizer"]=(path,kw);return "tokenizer"
    def watermark(**kw):calls["watermark"]=kw;return "watermark"
    modules={"generator":generator,"silentcipher":NS(get_model=watermark),"transformers":NS(AutoTokenizer=NS(from_pretrained=tokenizer))}
    oldpath=sys.path[:]
    try:
        with patch.dict(sys.modules,modules):
            assert worker.load_generator(root,root/"engine",cfg,[1,2,3,4,5],torch)=="loaded"
            assert generator.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")=="tokenizer"
            refused(lambda:generator.AutoTokenizer.from_pretrained("other"),"assets_corrupt")
            assert generator.hf_hub_download("kyutai/moshiko-pytorch-bf16","tokenizer-e351c8d8-checkpoint125.safetensors")==str(root/"mimi/tokenizer-e351c8d8-checkpoint125.safetensors")
            refused(lambda:generator.hf_hub_download("other","file"),"assets_corrupt")
            assert generator.load_watermarker("cpu")=="watermark"
        assert calls["tokenizer"]==(str(root/"tokenizer"),{"local_files_only":True,"trust_remote_code":False})
        assert calls["load"]["model_path_or_repo_id"]==str(root/"weights/model.safetensors")
        assert calls["watermark"]["ckpt_path"]==str(root/"watermarker")
        assert generator.MISO_TTS_WATERMARK==[1,2,3,4,5]
    finally:sys.path[:]=oldpath


def parent_death(root):
    pidfile=root/"orphan-pid"
    code="\n".join([
        "import sys,time,types;from pathlib import Path",
        "p=types.ModuleType('owned');p.__path__=[sys.argv[1]];sys.modules[p.__name__]=p",
        "from owned.process import run_process",
        "child=\"import os,time;from pathlib import Path;Path(__import__('sys').argv[1]).write_text(str(os.getpid()));time.sleep(30)\"",
        "run_process([sys.executable,'-B','-c',child,sys.argv[2]],directory=Path(sys.argv[2]).parent,deadline=time.monotonic()+30,cancel=lambda:False,limit=1024)"])
    parent=subprocess.Popen([sys.executable,"-E","-B","-c",code,str(Path(logic.__file__).parent),str(pidfile)])
    child=None
    try:
        wait(pidfile.is_file);child=int(pidfile.read_text());parent.kill();parent.wait(timeout=3)
        wait(lambda:not alive(child),3)
    finally:
        if parent.poll() is None:parent.kill();parent.wait(timeout=3)
        if child and alive(child):os.kill(child,signal.SIGKILL)


def main():
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (AssetTests,ListenerTests))
    result=unittest.TextTestRunner().run(suite);assert result.wasSuccessful()
    cfg=logic.configuration({})
    for value in ({"temperature":float("nan")},{"threads":True},{"max_pending":17},{"device":[]},{"unknown":"x"},{"watermark_key_ref":"[1,2,3,4,5]"},{"asset_directory":"\ud800"}):
        refused(lambda:logic.configuration(value),"config")
    for value in (None,"",{"text":"x"},{"text":"x","request_id":"../x"},"\ud800","x"*1001):
        refused(lambda:logic.request(value,cfg),"input")
    assert logic.request('{"action":"interrupt"}',cfg)["text"]=='{"action":"interrupt"}'
    for value in ('{"action":"interrupt","action":"interrupt"}',{"action":"stop"},{"action":"interrupt","extra":1}):
        refused(lambda:logic.interrupt(value),"input")
    assert logic.interrupt('{"action":"interrupt"}')=={"action":"interrupt"}
    for value in ("[1,2,3,4,true]","[256,2,3,4,5]","[1,2]",None):
        refused(lambda:logic.watermark_key(lambda _:value,"ref"),"credentials")
    with TemporaryDirectory() as directory:
        root=Path(directory);storage=root/"storage";storage.mkdir();cfg=logic.configuration(settings(root));adapter_tests(root)
        os.environ["MISO_TEST_SENTINEL"]="private"
        try:
            # Execute the actual owned worker too: its dependency preflight must
            # fail cleanly on an interpreter without the separate Miso environment.
            import importlib.util
            if importlib.util.find_spec("torch") is None:
                refused(lambda:logic.generate({"text":"Hello","request_id":"real-preflight"},
                    {**cfg,"environment_python":sys.executable},root,storage,lambda _:SECRET),"dependency")
            files=[]
            for fmt in ("wav","ogg_opus"):
                prepared=logic.generate({"text":"Hello","request_id":fmt},{**cfg,"format":fmt},root,storage,lambda _:SECRET)
                assert not (storage/"audio").exists() if not files else len(list((storage/"audio").iterdir()))==len(files)
                try:
                    metadata=logic.publish(prepared);files.append(check_file(metadata))
                finally:prepared.close()
            for text,code in (("fixture:invalid","audio_invalid"),("fixture:budget","audio_limit"),("fixture:error","model_failed")):
                refused(lambda:logic.generate({"text":text,"request_id":code},cfg,root,storage,lambda _:SECRET),code)
            cancelled=threading.Event();results=[]
            def job():
                try:logic.generate({"text":"fixture:hold","request_id":"cancel"},cfg,root,storage,lambda _:SECRET,cancelled.is_set)
                except Exception as exc:results.append(exc)
            thread=threading.Thread(target=job);thread.start();pid=started(cfg,"cancel")["pid"];cancelled.set();thread.join(5)
            assert not thread.is_alive() and results[0].code=="cancelled";wait(lambda:not alive(pid))
            refused(lambda:logic.generate({"text":"fixture:hold","request_id":"timeout"},{**cfg,"timeout_sec":1},root,storage,lambda _:SECRET),"timeout")
            prepared=logic.generate({"text":"Hello","request_id":"late"},cfg,root,storage,lambda _:SECRET)
            try:refused(lambda:logic.publish(prepared,lambda:True),"cancelled")
            finally:prepared.close()
            assert set((storage/"audio").iterdir())==set(files)
            assert not list(storage.glob("miso-job-*")) and not list((storage/"audio").glob(".miso*"))
            for command,limit,code in (("print('x'*10000)",10,"process_output"),("import time;time.sleep(3)",100,"timeout")):
                refused(lambda:logic.run_process([sys.executable,"-B","-c",command],directory=root,deadline=time.monotonic()+.4,cancel=lambda:False,limit=limit),code)
            parent_death(root)
        finally:os.environ.pop("MISO_TEST_SENTINEL",None)
    print("[ok] Miso adapter contracts, real process fencing/codecs/publication and 11 asset/listener tests; model inference simulated",flush=True)


if __name__=="__main__":main()
