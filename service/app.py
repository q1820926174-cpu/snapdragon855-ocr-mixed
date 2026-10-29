"""HTTP API for cropped-text recognition. No URL fetching or image persistence."""
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import secrets

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .backend import Backend, Config, Busy, Deadline, InvalidImage, ImageTooLarge, Unavailable


class BodyLimit:
    def __init__(self, app, limit): self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http': return await self.app(scope, receive, send)
        if scope['method'] not in ['POST','PUT','PATCH']:
            return await self.app(scope, receive, send)
        total, chunks = 0, []
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect': return
            if message['type'] == 'http.request':
                total += len(message.get('body', b''))
                if total > self.limit:
                    return await JSONResponse({'detail':'Request body too large / 请求体过大'}, status_code=413)(scope, receive, send)
                chunks.append(message.get('body', b''))
                if not message.get('more_body', False): break
        body, delivered = b''.join(chunks), False
        async def bounded_receive():
            nonlocal delivered
            if delivered: return await receive()
            delivered = True
            return {'type':'http.request','body':body,'more_body':False}
        await self.app(scope, bounded_receive, send)


class BatchItem(BaseModel):
    id: str = Field(default='', max_length=128)
    image_base64: str = Field(min_length=1, max_length=7*1024*1024)


class Batch(BaseModel):
    images: list[BatchItem] = Field(min_length=1, max_length=32)
    auto_rotate: bool = True


def create_app(config=None):
    config = config or Config.environment()

    @asynccontextmanager
    async def lifespan(app):
        backend = Backend(config)
        app.state.backend = backend
        await backend.start()
        try: yield
        finally: await backend.close()

    app = FastAPI(title='Snapdragon 855 OCR / 骁龙 855 文字识别', version='0.3.0', lifespan=lifespan,
                  description='GPU recognition for cropped text lines. 文字裁剪图识别，不包含整页检测。')
    app.add_middleware(BodyLimit, limit=config.max_body_bytes)
    bearer = HTTPBearer(auto_error=False, description='Optional when OCR_API_KEY is empty / 未配置 OCR_API_KEY 时可省略')

    async def authorize(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if config.api_key:
            actual=request.headers.get('authorization','').encode()
            expected=('Bearer '+config.api_key).encode()
            if not secrets.compare_digest(actual, expected):
                raise HTTPException(401, 'Invalid API key / API 密钥无效', headers={'WWW-Authenticate':'Bearer'})

    async def results(payloads, auto_rotate):
        try:
            tasks = app.state.backend.submit(payloads, auto_rotate)
            # A disconnected caller cannot release GPU capacity before its work ends.
            group = asyncio.gather(*tasks, return_exceptions=True)
            values = await asyncio.shield(group)
            for value in values:
                if isinstance(value, BaseException): raise value
            return values
        except Busy as exc: raise HTTPException(429, str(exc), headers={'Retry-After':'1'}) from exc
        except ImageTooLarge as exc: raise HTTPException(413, str(exc)) from exc
        except InvalidImage as exc: raise HTTPException(422, str(exc)) from exc
        except Deadline as exc: raise HTTPException(504, str(exc)) from exc
        except Unavailable as exc: raise HTTPException(503, str(exc)) from exc

    @app.get('/healthz', summary='Liveness / 存活检查')
    async def health(): return {'status':'alive'}

    @app.get('/readyz', summary='GPU readiness / GPU 就绪检查')
    async def ready():
        if not app.state.backend.ready: raise HTTPException(503, 'GPU backend unavailable')
        return {'status':'ready', 'backend':'vulkan', 'model':'ppocrv6-medium-sm8150-gpu'}

    @app.get('/v1/models', dependencies=[Depends(authorize)], summary='Available model / 可用模型')
    async def models():
        return {'data':[{'id':'ppocrv6-medium-sm8150-gpu', 'task':'text-recognition',
                         'weights_release':'v0.2.0',
                         'backend':'vulkan', 'cnn':'fp16-packed/fp16-arithmetic', 'tail':'fp32',
                         'input':[1,48,640,3], 'cpu_fallback':False, 'npu':False}]}

    @app.post('/v1/recognize', dependencies=[Depends(authorize)], summary='One cropped text image / 单张文字裁剪图识别',
              openapi_extra={'requestBody':{'required':True,'content':{'image/png':{'schema':{'type':'string','format':'binary'}},'image/jpeg':{'schema':{'type':'string','format':'binary'}}}}})
    async def recognize(request: Request, auto_rotate: bool = True):
        data = await request.body()
        if len(data)>config.max_image_bytes: raise HTTPException(413,'Image exceeds size limit')
        return (await results([data], auto_rotate))[0]

    @app.post('/v1/recognize/batch', dependencies=[Depends(authorize)], summary='Ordered batch of up to 32 crops / 最多 32 张批量识别，按输入顺序返回')
    async def batch(body: Batch):
        payloads=[]
        for image in body.images:
            try: data=base64.b64decode(image.image_base64, validate=True)
            except (binascii.Error, ValueError) as exc: raise HTTPException(422,'Invalid base64 image') from exc
            if not data or len(data)>config.max_image_bytes: raise HTTPException(413,'Image exceeds size limit')
            payloads.append(data)
        values=await results(payloads,body.auto_rotate)
        return {'results':[dict(value,id=image.id) for image,value in zip(body.images,values)]}

    @app.get('/v1/stats', dependencies=[Depends(authorize)], summary='Runtime counters / 运行计数')
    async def stats(): return app.state.backend.stats()

    @app.get('/metrics', dependencies=[Depends(authorize)], response_class=PlainTextResponse)
    async def metrics():
        stats=app.state.backend.stats()
        return ''.join(f'ocr_{key} {stats[key]}\n' for key in ['pending','completed','rejected','failed','gpu_ms_total'])

    return app
