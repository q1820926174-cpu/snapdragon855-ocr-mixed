import base64
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import sys,time,unittest

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from service.app import create_app
from service.backend import Config,image_tensor,InvalidImage


def config(*args,**kw):
    return Config(command=(sys.executable,str(Path(__file__).with_name('worker_fixture.py')),*args),exit_on_failure=False,**kw)


def image():
    buffer=io.BytesIO();Image.new('RGB',(40,12),'red').save(buffer,format='PNG');return buffer.getvalue()


class ApiContracts(unittest.TestCase):
    def test_images_batch_order_model_reuse_and_auth(self):
        with TestClient(create_app(config(api_key='secret'))) as client:
            self.assertEqual(client.get('/readyz').status_code,200)
            self.assertEqual(client.post('/v1/recognize',content=image()).status_code,401)
            headers={'Authorization':'Bearer secret'}
            one=client.post('/v1/recognize',content=image(),headers=headers)
            self.assertEqual(one.status_code,200,one.text);self.assertEqual(one.json()['text'],'fixture')
            encoded=base64.b64encode(image()).decode()
            response=client.post('/v1/recognize/batch',json={'images':[{'id':v,'image_base64':encoded} for v in ['b','a']]},headers=headers)
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual([v['id'] for v in response.json()['results']],['b','a'])
            stats=client.get('/v1/stats',headers=headers).json()
            self.assertEqual(stats['completed'],3);self.assertEqual(stats['pending'],0)
            self.assertEqual(stats['worker']['model_initializations'],1)
            self.assertFalse(client.get('/v1/models',headers=headers).json()['data'][0]['cpu_fallback'])
            self.assertIn('HTTPBearer',client.get('/openapi.json').json()['components']['securitySchemes'])

    def test_invalid_and_oversized_requests_do_not_reach_worker(self):
        with TestClient(create_app(config(max_image_bytes=512,max_body_bytes=1024))) as client:
            self.assertEqual(client.post('/v1/recognize',content=b'invalid').status_code,422)
            self.assertEqual(client.post('/v1/recognize',content=b'x'*513).status_code,413)
            self.assertEqual(client.post('/v1/recognize',content=iter([b'x'*600,b'y'*600])).status_code,413)
            self.assertEqual(client.post('/v1/recognize/batch',json={'images':[{'image_base64':'!'}]}).status_code,422)
            self.assertEqual(client.post('/v1/recognize/batch',json={'images':[]}).status_code,422)
            self.assertEqual(client.get('/v1/stats').json()['completed'],0)

    def test_queue_rejects_instead_of_reinitializing_or_growing(self):
        with TestClient(create_app(config('0.4',workers=1,inflight=1,pending=1))) as client, ThreadPoolExecutor(1) as pool:
            running=pool.submit(client.post,'/v1/recognize',content=image())
            for _ in range(100):
                if client.get('/v1/stats').json()['pending']:break
                time.sleep(.01)
            response=client.post('/v1/recognize',content=image())
            self.assertEqual(response.status_code,429,response.text)
            self.assertEqual(running.result(timeout=5).status_code,200)
            self.assertEqual(client.get('/v1/stats').json()['worker']['model_initializations'],1)

    def test_dead_backend_fails_closed(self):
        with TestClient(create_app(config('exit'))) as client:
            self.assertEqual(client.post('/v1/recognize',content=image()).status_code,503)
            self.assertEqual(client.get('/readyz').status_code,503)

    def test_pixel_limit_and_tall_rotation(self):
        buf=io.BytesIO();Image.new('RGB',(10,40),'red').save(buf,format='PNG')
        array=np.frombuffer(image_tensor(buf.getvalue(),config(),True),'<f4').reshape(48,640,3)
        self.assertTrue((array[:,100,2]==1).all())
        with self.assertRaises(InvalidImage):image_tensor(buf.getvalue(),Config(max_pixels=100),True)
        with TestClient(create_app(config(max_pixels=100))) as client:
            self.assertEqual(client.post('/v1/recognize',content=image()).status_code,413)


if __name__=='__main__':unittest.main()
