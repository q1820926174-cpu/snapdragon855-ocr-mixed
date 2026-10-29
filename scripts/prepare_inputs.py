"""Recognition crops only: normalized NHWC BGR float32, no text detection."""
from pathlib import Path
import argparse,math,json,hashlib
import numpy as np
from PIL import Image

def preprocess(image):
    image=image.convert('RGB');width=min(640,math.ceil(48*image.width/image.height));out=np.zeros((48,640,3),np.float32)
    out[:,:width]=np.asarray(image.resize((width,48),Image.Resampling.BILINEAR),np.float32)[:,:,::-1]/127.5-1
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('images',type=Path);p.add_argument('output',type=Path);p.add_argument('--device-prefix',required=True,help='Absolute path where raw files will be copied on the phone');a=p.parse_args()
    if not a.device_prefix.startswith('/') or any(c in a.device_prefix for c in '\r\n\t'):p.error('Use an absolute Linux path without control characters')
    files=sorted(path for path in a.images.iterdir() if path.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'});assert files,'No image crops';a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for i,path in enumerate(files):
        ident='crop'+str(i).zfill(5);preprocess(Image.open(path)).tofile(a.output/(ident+'.raw'));rows.append(ident+'\t'+a.device_prefix.rstrip('/')+'/'+ident+'.raw')
    (a.output/'input.tsv').write_bytes(('\n'.join(rows)+'\n').encode('utf8'));print('PREPARED',len(rows),'crops')
