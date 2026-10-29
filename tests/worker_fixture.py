"""CPU-only transport fixture for tests. This is not an inference backend."""
import json,struct,sys,time
print(json.dumps(dict(type='ready',protocol=1,backend='vulkan',model_initializations=1,fixture=True)),flush=True)
while header:=sys.stdin.buffer.read(8):
    ident=struct.unpack('<Q',header)[0]
    data=sys.stdin.buffer.read(368640)
    if len(data)!=368640:sys.exit(65)
    if sys.argv[1:] == ['exit']:sys.exit(7)
    if sys.argv[1:]:time.sleep(float(sys.argv[1]))
    print(json.dumps(dict(type='result',id=ident,text='fixture',confidence=.99,hash='test',gpu_ms=1)),flush=True)
