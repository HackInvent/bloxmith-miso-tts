"""Bounded complete-file synthesis; publication is separate from cancellable native work."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
from uuid import uuid4
import wave
from .process import MisoError, Cancelled, check, run_process

DEFAULTS = {
    "asset_directory": "", "environment_python": "", "output_directory": "",
    "watermark_key_ref": "", "device": "cuda:0", "precision": "bfloat16",
    "format": "ogg_opus", "temperature": 0.9, "topk": 50, "threads": 2,
    "max_audio_sec": 30, "max_text_chars": 1000, "timeout_sec": 600,
    "memory_limit_mib": 98304, "min_available_ram_mib": 65536, "max_pending": 4,
}


def configuration(raw):
    if not isinstance(raw, Mapping) or set(raw)-set(DEFAULTS)-{"execution","position","runtime_path","runtime_path_label"}:
        raise MisoError("config")
    cfg={**DEFAULTS,**{k:v for k,v in raw.items() if k in DEFAULTS}}
    for key in ("asset_directory","environment_python","output_directory","watermark_key_ref"):
        value=cfg[key]
        try:
            if not isinstance(value,str) or len(value.encode("utf-8"))>4096 or any(ord(c)<32 for c in value):
                raise MisoError("config")
        except UnicodeError:
            raise MisoError("config") from None
    if cfg["watermark_key_ref"] and not re.fullmatch(r"(?:secret://(?:workspace|project)/)?[A-Za-z0-9_.-]{1,128}",cfg["watermark_key_ref"]):
        raise MisoError("config")
    for key,choices in (("device",("cpu","cuda:0")),("precision",("float32","bfloat16")),("format",("wav","ogg_opus"))):
        if not isinstance(cfg[key],str) or cfg[key] not in choices:
            raise MisoError("config")
    for key,low,high in (("topk",1,500),("threads",1,16),("max_audio_sec",1,90),
            ("max_text_chars",1,2000),("timeout_sec",1,3600),("memory_limit_mib",4096,262144),
            ("min_available_ram_mib",16384,262144),("max_pending",0,16)):
        if type(cfg[key]) is not int or not low<=cfg[key]<=high:
            raise MisoError("config")
    if type(cfg["temperature"]) not in (int,float) or not math.isfinite(cfg["temperature"]) or not .1<=cfg["temperature"]<=2:
        raise MisoError("config")
    return cfg


def json_value(raw, limit):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise MisoError("input")
            out[k]=v
        return out
    if not isinstance(raw,str): return raw
    try:
        if len(raw.encode("utf-8"))>limit: raise MisoError("input")
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(MisoError("input")))
    except (ValueError,UnicodeError,RecursionError):
        raise MisoError("input") from None


def interrupt(raw):
    value=json_value(raw,1024)
    if not isinstance(value,Mapping) or dict(value)!={"action":"interrupt"}:
        raise MisoError("input")
    return {"action":"interrupt"}


def request(raw, cfg):
    correlation=uuid4().hex
    # Plain strings remain speech, even if they look like commands. Correlated
    # JSON must arrive as a mapping, or be decoded according to its input MIME.
    if isinstance(raw,Mapping):
        if set(raw)!={"text","request_id"}: raise MisoError("input")
        correlation=raw["request_id"];raw=raw["text"]
        if not isinstance(correlation,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}",correlation):
            raise MisoError("input")
    try:
        if not isinstance(raw,str) or not raw.strip() or len(raw)>cfg["max_text_chars"] or len(raw.encode("utf-8"))>8192 or "\0" in raw:
            raise MisoError("input")
    except UnicodeError:
        raise MisoError("input") from None
    return {"text":raw.strip(),"request_id":correlation}


def watermark_key(resolver, reference):
    if not reference or not callable(resolver): raise MisoError("credentials")
    try:
        raw=resolver(reference)
        if not isinstance(raw,str) or len(raw)>128: raise ValueError()
        value=json.loads(raw)
        if not isinstance(value,list) or len(value)!=5 or any(type(v) is not int or not 0<=v<=255 for v in value):
            raise ValueError()
        return value
    except Exception:
        raise MisoError("credentials") from None


def resolve_path(root, value):
    path=Path(value).expanduser()
    return (Path(root)/path).absolute() if not path.is_absolute() else path


@dataclass
class PreparedAudio:
    owner: object
    path: Path
    output_directory: Path
    metadata: dict
    deadline: float

    def close(self):
        self.owner.cleanup()


def audio_info(path, cfg, raw):
    try:
        info=json_value(raw.decode("utf-8") if isinstance(raw,bytes) else raw,8192)
        if set(info)!={"complete","watermarked","sample_rate","frames","gain"} or info["complete"] is not True or info["watermarked"] is not True:
            raise ValueError()
        if type(info["frames"]) is not int or not 0<info["frames"]<=24000*cfg["max_audio_sec"] or type(info["sample_rate"]) is not int or info["sample_rate"]!=24000:
            raise ValueError()
        if type(info["gain"]) not in (int,float) or not math.isfinite(info["gain"]) or not 0<info["gain"]<=1:
            raise ValueError()
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as source:
            file=os.fstat(source.fileno())
            if not stat.S_ISREG(file.st_mode) or not 44<=file.st_size<=24000*cfg["max_audio_sec"]*2+128:
                raise ValueError()
            with wave.open(source,"rb") as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes())!=(1,2,24000,info["frames"]):
                    raise ValueError()
                if len(wav.readframes(info["frames"]+1))!=info["frames"]*2:
                    raise ValueError()
        return info
    except (ValueError,TypeError,KeyError,OSError,EOFError,wave.Error):
        raise MisoError("audio_invalid") from None


def generate(item, cfg, root, storage, resolver, cancel=lambda:False):
    cfg=configuration(cfg);item=request(item,cfg)
    storage=Path(storage)
    if not storage.is_absolute() or storage.is_symlink() or not storage.is_dir(): raise MisoError("storage")
    assets=resolve_path(root,cfg["asset_directory"])
    if not cfg["asset_directory"] or not assets.is_dir() or assets.is_symlink(): raise MisoError("assets_missing")
    python=resolve_path(root,cfg["environment_python"]) if cfg["environment_python"] else Path(sys.executable)
    if not python.is_file() or not os.access(python,os.X_OK): raise MisoError("dependency")
    target=resolve_path(root,cfg["output_directory"]) if cfg["output_directory"] else storage/"audio"
    if target.is_symlink() or target.exists() and not target.is_dir(): raise MisoError("storage")
    key=watermark_key(resolver,cfg["watermark_key_ref"])
    started=time.monotonic();deadline=started+cfg["timeout_sec"]
    owner=tempfile.TemporaryDirectory(prefix="miso-job-",dir=storage);job=Path(owner.name)
    try:
        envelope={"config":cfg,"request":item,"assets":str(assets),"job":str(job),"watermark":key}
        raw=run_process([str(python),"-E","-B",str(Path(__file__).with_name("worker.py"))],directory=job,
            deadline=deadline,cancel=cancel,limit=8192,stdin=json.dumps(envelope,ensure_ascii=True).encode())
        del envelope,key
        wav=job/"generated.wav";info=audio_info(wav,cfg,raw)
        output=wav
        if cfg["format"]=="ogg_opus":
            output=job/"generated.ogg"
            run_process(["ffmpeg","-v","error","-nostdin","-n","-threads","1","-protocol_whitelist","file,pipe",
                "-format_whitelist","wav","-i",str(wav),"-map_metadata","-1","-ac","1","-ar","48000",
                "-c:a","libopus","-b:a","96000","-f","ogg",str(output)],directory=job,deadline=deadline,cancel=cancel,limit=1024)
        check(deadline,cancel)
        info.update(request_id=item["request_id"],duration_sec=info["frames"]/24000,
            source_frames=info["frames"],
            format=cfg["format"],sample_rate=48000 if cfg["format"]=="ogg_opus" else 24000,channels=1,
            source_sample_rate=24000,elapsed_sec=round(time.monotonic()-started,4),
            model="MisoLabs/MisoTTS",model_revision="ef6b096cc35d3cde6aa0721013648416c14c36b2",
            engine_revision="cd7dde13dbfec33fd8321e6c34a43f86724a8bad",
            settings={k:cfg[k] for k in ("device","precision","temperature","topk","max_audio_sec")})
        del info["frames"]
        return PreparedAudio(owner,output,target,info,deadline)
    except BaseException:
        owner.cleanup()
        raise


def publish(prepared, cancel=lambda:False):
    """Publish after the listener has drained pending interrupts; never overwrite a file."""
    check(prepared.deadline,cancel)
    target=prepared.output_directory
    if target.is_symlink(): raise MisoError("storage")
    target.mkdir(mode=0o700,parents=True,exist_ok=True)
    suffix=".ogg" if prepared.metadata["format"]=="ogg_opus" else ".wav"
    final=target/("miso-"+uuid4().hex+suffix)
    temporary=None
    try:
        fd=os.open(prepared.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as source:
            size=os.fstat(source.fileno())
            if not stat.S_ISREG(size.st_mode) or not 0<size.st_size<=33554432: raise MisoError("audio_invalid")
            descriptor,temporary=tempfile.mkstemp(prefix=".miso-publish-",dir=target)
            digest=hashlib.sha256();total=0
            with os.fdopen(descriptor,"wb") as output:
                while chunk:=source.read(65536):
                    check(prepared.deadline,cancel);total+=len(chunk)
                    if total>33554432: raise MisoError("audio_invalid")
                    output.write(chunk);digest.update(chunk)
                output.flush();os.fsync(output.fileno())
            if total!=size.st_size: raise MisoError("audio_invalid")
        check(prepared.deadline,cancel)
        os.link(temporary,final,follow_symlinks=False)
        metadata={**prepared.metadata,"path":str(final),"bytes":total,"sha256":digest.hexdigest(),
            "content_type":"audio/ogg" if suffix==".ogg" else "audio/wav"}
        return metadata
    except OSError:
        raise MisoError("storage") from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
