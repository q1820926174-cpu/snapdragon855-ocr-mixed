"""One persistent native GPU process; bounded admission and no per-image files."""
import asyncio
from dataclasses import dataclass
import io
import itertools
import json
import os
from pathlib import Path
import struct
import sys
import time
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_inputs import preprocess


class Unavailable(Exception): pass
class Busy(Exception): pass
class InvalidImage(Exception): pass
class ImageTooLarge(InvalidImage): pass
class Deadline(Exception): pass


@dataclass
class Config:
    command: tuple = ('/usr/local/bin/pnnx-gpu-worker',)
    models: str = '/models'
    workers: int = 2
    inflight: int = 10
    pending: int = 32
    timeout: float = 30
    max_image_bytes: int = 5 * 1024 * 1024
    max_body_bytes: int = 16 * 1024 * 1024
    max_pixels: int = 20_000_000
    api_key: str = ''
    exit_on_failure: bool = True

    def __post_init__(self):
        if not 1 <= self.workers <= 4 or not self.workers <= self.inflight <= 32 or not self.inflight <= self.pending <= 128:
            raise ValueError('Require 1..4 workers, at most 32 inflight and at most 128 pending')
        if not 1 <= self.timeout <= 300 or not 1 <= self.max_image_bytes <= self.max_body_bytes <= 64 * 1024 * 1024:
            raise ValueError('Invalid request size or timeout limits')

    @classmethod
    def environment(cls):
        return cls(command=(os.getenv('OCR_WORKER_BINARY', '/usr/local/bin/pnnx-gpu-worker'),),
                   models=os.getenv('OCR_MODEL_DIR', '/models'),
                   workers=int(os.getenv('OCR_GPU_WORKERS', '2')),
                   inflight=int(os.getenv('OCR_MAX_INFLIGHT', '10')),
                   pending=int(os.getenv('OCR_MAX_PENDING', '32')),
                   timeout=float(os.getenv('OCR_REQUEST_TIMEOUT', '30')),
                   api_key=os.getenv('OCR_API_KEY', ''))


def image_tensor(data, config, auto_rotate):
    if not data: raise InvalidImage('Empty image')
    if len(data) > config.max_image_bytes: raise ImageTooLarge('Image exceeds encoded size limit')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > config.max_pixels:
                    raise ImageTooLarge('Image exceeds pixel limit')
                image = ImageOps.exif_transpose(image).convert('RGB')
                if auto_rotate and image.height / image.width > 1.5:
                    image = image.transpose(Image.Transpose.ROTATE_270)
                return preprocess(image).astype('<f4', copy=False).tobytes()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLarge('Image exceeds pixel limit') from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InvalidImage('Invalid image') from exc


class Backend:
    def __init__(self, config):
        self.config = config
        self.proc = None
        self.info = {}
        self.failure = None
        self.closing = False
        self.pending = self.completed = self.rejected = self.failed = 0
        self.total_gpu_ms = 0.0
        self.started = time.monotonic()
        self.sequence = itertools.count(1)
        self.futures = {}
        self.jobs = set()
        self.slots = asyncio.Semaphore(config.inflight)
        self.writer = asyncio.Lock()

    @property
    def ready(self):
        return bool(self.proc and self.proc.returncode is None and self.info and not self.failure and not self.closing)

    async def start(self):
        env = dict(os.environ, SCV41_PIPE_BASE=self.config.models,
                   OCR_GPU_WORKERS=str(self.config.workers), OCR_MAX_INFLIGHT=str(self.config.inflight),
                   SCV41_NCNN_SAFE_GEMM='1', SCV41_NCNN_FP32_SLM64='1', SCV41_NCNN_STRICT_GPU='1')
        env.pop('SCV41_PNNX_SUBGROUP64', None)
        env.pop('SCV41_GPU_CTC_DEBUG', None)
        env.pop('SCV41_NCNN_NODE_SYNC', None)
        self.proc = await asyncio.create_subprocess_exec(*self.config.command, env=env,
                          stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        try:
            self.info = json.loads(await asyncio.wait_for(self.proc.stdout.readline(), 120))
            if self.info.get('type') != 'ready' or self.info.get('backend') != 'vulkan' or self.info.get('protocol') != 1:
                raise Unavailable('Unsupported native worker protocol')
        except BaseException:
            await self.close()
            raise
        self.reader = asyncio.create_task(self._read())

    def abort(self, reason):
        if self.failure or self.closing:
            return
        self.failure = reason
        for future in list(self.futures.values()):
            if not future.done(): future.set_exception(Unavailable(reason))
        if self.proc and self.proc.returncode is None:
            try: self.proc.terminate()
            except ProcessLookupError: pass
        if self.config.exit_on_failure:
            # Docker restart policy recovers a failed GPU worker; no silent CPU mode.
            asyncio.get_running_loop().call_later(0.5, os._exit, 70)

    async def _read(self):
        try:
            while line := await self.proc.stdout.readline():
                result = json.loads(line)
                if result.get('type') != 'result': raise Unavailable('Invalid worker response')
                future = self.futures.pop(result['id'], None)
                if future is None: raise Unavailable('Unknown worker request id')
                if not future.done(): future.set_result(result)
            if not self.closing: self.abort('GPU worker exited')
        except Exception:
            self.abort('GPU worker protocol failed')

    async def close(self):
        self.closing = True
        for job in list(self.jobs): job.cancel()
        if self.jobs: await asyncio.gather(*list(self.jobs), return_exceptions=True)
        if self.proc:
            if self.proc.stdin: self.proc.stdin.close()
            try: await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.proc.kill(); await self.proc.wait()
        reader = getattr(self, 'reader', None)
        if reader:
            reader.cancel(); await asyncio.gather(reader, return_exceptions=True)

    def submit(self, images, auto_rotate=True):
        if not self.ready: raise Unavailable('GPU backend is not ready')
        if self.pending + len(images) > self.config.pending:
            self.rejected += len(images)
            raise Busy('Inference queue is full')
        self.pending += len(images)
        tasks = []
        for data in images:
            task = asyncio.create_task(self._infer(data, auto_rotate))
            self.jobs.add(task)
            task.add_done_callback(self.jobs.discard)
            tasks.append(task)
        return tasks

    async def _infer(self, data, auto_rotate):
        ident = next(self.sequence)
        sent = False
        start = time.monotonic()
        try:
            async with asyncio.timeout(self.config.timeout):
                async with self.slots:
                    raw = await asyncio.to_thread(image_tensor, data, self.config, auto_rotate)
                    async with self.writer:
                        if not self.ready: raise Unavailable('GPU backend is not ready')
                        future = asyncio.get_running_loop().create_future()
                        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
                        self.futures[ident] = future
                        sent = True
                        self.proc.stdin.write(struct.pack('<Q', ident) + raw)
                        await self.proc.stdin.drain()
                    result = await asyncio.shield(future)
                    self.completed += 1
                    self.total_gpu_ms += result['gpu_ms']
                    return dict(text=result['text'], confidence=result['confidence'], hash=result['hash'],
                                gpu_ms=result['gpu_ms'], total_ms=(time.monotonic()-start)*1000,
                                backend='vulkan', model='ppocrv6-medium-sm8150-gpu')
        except TimeoutError as exc:
            self.failed += 1
            if sent: self.abort('GPU inference deadline exceeded')
            raise Deadline('Inference deadline exceeded') from exc
        except (BrokenPipeError, ConnectionError) as exc:
            self.failed += 1; self.abort('GPU worker pipe failed')
            raise Unavailable('GPU worker pipe failed') from exc
        except Exception:
            self.failed += 1
            raise
        finally:
            self.pending -= 1
            self.futures.pop(ident, None)

    def stats(self):
        return dict(ready=self.ready, pending=self.pending, completed=self.completed,
                    rejected=self.rejected, failed=self.failed, gpu_ms_total=self.total_gpu_ms,
                    max_pending=self.config.pending, max_inflight=self.config.inflight,
                    uptime_seconds=time.monotonic()-self.started, worker=self.info)
