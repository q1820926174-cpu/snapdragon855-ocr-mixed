"""Concurrent raw-image API client, standard library only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json,mimetypes,os,sys
from pathlib import Path
import urllib.request


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser(description='Recognize cropped text images / 识别文字裁剪图')
    parser.add_argument('images',nargs='+',type=Path)
    parser.add_argument('--url',default='http://127.0.0.1:18081')
    parser.add_argument('--concurrency',type=int,default=10)
    args=parser.parse_args()
    if not 1<=args.concurrency<=32:parser.error('concurrency must be 1..32')
    def recognize(path):
        headers={'Content-Type':mimetypes.guess_type(path.name)[0] or 'application/octet-stream'}
        if os.getenv('OCR_API_KEY'):headers['Authorization']='Bearer '+os.environ['OCR_API_KEY']
        req=urllib.request.Request(args.url.rstrip('/')+'/v1/recognize',data=path.read_bytes(),headers=headers)
        with urllib.request.urlopen(req,timeout=35) as response:return dict(file=str(path),**json.load(response))
    with ThreadPoolExecutor(args.concurrency) as executor:
        for result in executor.map(recognize,args.images):print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
