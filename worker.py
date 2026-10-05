"""Private offline worker for the checksum-pinned official Miso implementation."""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import resource
import sys
import types
import wave

if __package__:
    from .assets_store import AssetError, prepare
    from .logic import configuration
    from .process import MisoError
else:
    package=types.ModuleType("miso_worker_owned")
    package.__path__=[str(Path(__file__).resolve().parent)];sys.modules[package.__name__]=package
    from miso_worker_owned.assets_store import AssetError, prepare
    from miso_worker_owned.logic import configuration
    from miso_worker_owned.process import MisoError


def available_ram():
    values={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()}
    available=int(values["MemAvailable"].split()[0])*1024
    # A smaller cgroup v2 allowance wins over host memory. Unknown limits are
    # not resource reservations; the native address-space limit remains active.
    try:
        cap=Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if cap!="max":
            used=int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            available=min(available,max(0,int(cap)-used))
    except (OSError,ValueError):
        pass
    return available


def guard_eos(model, torch):
    """Observe the official all-zero stop frame; never accept frame-budget exhaustion."""
    original=model.generate_frame
    state={"ended":False,"frames":0}
    def generate(*args,**kwargs):
        result=original(*args,**kwargs)
        if tuple(result.shape)!=(1,32) or state["ended"]:
            raise MisoError("model_failed")
        if bool(torch.all(result==0)):
            state["ended"]=True
        else:
            state["frames"]+=1
        return result
    model.generate_frame=generate
    return state


def load_generator(assets,engine,cfg,key,torch):
    """Bind upstream loaders to verified local paths without editing upstream files."""
    sys.path.insert(0,str(engine))
    import generator
    import silentcipher
    from transformers import AutoTokenizer
    tokenizer_name="meta-llama/Llama-3.2-1B"
    def tokenizer(name):
        if name!=tokenizer_name: raise MisoError("assets_corrupt")
        return AutoTokenizer.from_pretrained(str(assets/"tokenizer"),local_files_only=True,trust_remote_code=False)
    generator.AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer)
    def mimi(repo_id,filename):
        if (repo_id,filename)!=("kyutai/moshiko-pytorch-bf16","tokenizer-e351c8d8-checkpoint125.safetensors"):
            raise MisoError("assets_corrupt")
        return str(assets/"mimi"/filename)
    generator.hf_hub_download=mimi
    def watermarker(device):
        directory=assets/"watermarker"
        return silentcipher.get_model(model_type="44.1k",device=device,ckpt_path=str(directory),config_path=str(directory/"hparams.yaml"))
    generator.load_watermarker=watermarker
    generator.MISO_TTS_WATERMARK=list(key)
    return generator.load_miso_8b(device=cfg["device"],model_path_or_repo_id=str(assets/"weights"/"model.safetensors"),
        dtype=getattr(torch,cfg["precision"]))


def generate_audio(generator,torch,text,cfg,job):
    if generator.sample_rate!=24000: raise MisoError("model_failed")
    state=guard_eos(generator._model,torch)
    audio=generator.generate(text=text,speaker=0,context=[],max_audio_length_ms=cfg["max_audio_sec"]*1000,
        temperature=cfg["temperature"],topk=cfg["topk"])
    if not state["ended"]: raise MisoError("audio_limit")
    audio=audio.detach().to(device="cpu",dtype=torch.float32)
    if audio.ndim!=1 or not 0<audio.numel()<=24000*cfg["max_audio_sec"] or not bool(torch.isfinite(audio).all()):
        raise MisoError("audio_invalid")
    peak=float(audio.abs().max())
    if peak<=1e-7: raise MisoError("audio_invalid")
    gain=min(1.,.99/peak)
    pcm=(audio*gain*32767).round().to(torch.int16).numpy().astype("<i2",copy=False).tobytes()
    path=job/"generated.wav"
    with path.open("xb") as file:
        os.chmod(path,0o600)
        with wave.open(file,"wb") as output:
            output.setnchannels(1);output.setsampwidth(2);output.setframerate(24000);output.writeframes(pcm)
    return {"complete":True,"watermarked":True,"sample_rate":24000,"frames":audio.numel(),"gain":gain}


def require_dependencies():
    from importlib.metadata import distribution, version
    for name,expected in (("torch","2.4.0"),("torchaudio","2.4.0"),("tokenizers","0.21.0"),
            ("transformers","4.49.0"),("huggingface_hub","0.28.1"),("moshi","0.2.2"),
            ("torchtune","0.4.0"),("torchao","0.9.0")):
        if version(name).split("+")[0]!=expected: raise MisoError("dependency")
    try:
        origin=json.loads(distribution("silentcipher").read_text("direct_url.json") or "{}")
        if origin.get("url")!="https://github.com/SesameAILabs/silentcipher" or origin.get("vcs_info",{}).get("commit_id")!="d46d7d0893a583d8968ab3a6626e2289faec9152":
            raise ValueError()
    except (ValueError,TypeError,AttributeError):
        raise MisoError("dependency") from None


def infer(item):
    cfg=configuration(item["config"])
    limit=cfg["memory_limit_mib"]*1048576
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(cfg["timeout_sec"]*cfg["threads"],cfg["timeout_sec"]*cfg["threads"]+1))
    require_dependencies()
    if available_ram()<cfg["min_available_ram_mib"]*1048576: raise MisoError("resources")
    import torch
    torch.set_num_threads(cfg["threads"]);torch.set_num_interop_threads(1)
    if cfg["device"]=="cuda:0" and (not torch.cuda.is_available() or
            cfg["precision"]=="bfloat16" and not torch.cuda.is_bf16_supported()):
        raise MisoError("device")
    key=item["watermark"]
    if not isinstance(key,list) or len(key)!=5 or any(type(v) is not int or not 0<=v<=255 for v in key):
        raise MisoError("credentials")
    job=Path(item["job"]);assets,engine=prepare(item["assets"],job)
    generator=load_generator(assets,engine,cfg,key,torch)
    return generate_audio(generator,torch,item["request"]["text"],cfg,job)


def main():
    data=sys.stdin.buffer.read(32769)
    if len(data)>32768: raise MisoError("input")
    with redirect_stdout(sys.stderr):
        result=infer(json.loads(data))
    sys.stdout.write(json.dumps(result,ensure_ascii=True,allow_nan=False))


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        from importlib.metadata import PackageNotFoundError
        if isinstance(exc,MisoError): code=exc.code
        elif isinstance(exc,AssetError): code=str(exc)
        elif isinstance(exc,(ImportError,PackageNotFoundError)): code="dependency"
        elif isinstance(exc,MemoryError) or isinstance(exc,RuntimeError) and any(s in str(exc) for s in ("out of memory","DefaultCPUAllocator","Cannot allocate memory")):
            code="resources"
        else: code="model_failed"
        print(json.dumps({"error_code":code}));raise SystemExit(1) from None
