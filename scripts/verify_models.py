"""Verify open model assets before invoking device-side kernels."""
from pathlib import Path
import argparse,hashlib,json,struct

def verify_manifest(root):
    root=Path(root).resolve();manifest=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf8'))
    for name,expected in manifest.items():
        path=(root/name).resolve()
        if root not in path.parents:raise ValueError('Asset path escapes root')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Checksum mismatch: '+name)
    return len(manifest)

def verify_csr(path,co,ci,mean):
    import numpy as np
    data=Path(path).read_bytes()
    if len(data)<12:raise ValueError('Truncated CSR header')
    oc,ic,nnz=struct.unpack_from('<III',data)
    if (oc,ic)!=(co,ci) or not 0<nnz<=co*ci/20:raise ValueError('Invalid CSR dimensions or density')
    expected=12+4*(co+1+nnz*2+ci)
    if len(data)!=expected:raise ValueError('Invalid CSR byte count')
    words=np.frombuffer(data,dtype='<u4',offset=12);rows=words[:co+1];cols=words[co+1:co+1+nnz];values=words[co+1+nnz:co+1+2*nnz].view('<f4');means=words[-ci:].view('<f4')
    if rows[0]!=0 or rows[-1]!=nnz or np.any(rows[1:]<rows[:-1]) or np.any(cols>=ci):raise ValueError('Invalid CSR indices')
    if not np.isfinite(values).all() or not np.isfinite(means).all() or not np.array_equal(means,mean):raise ValueError('Invalid CSR values or mean')

def verify(root):
    import numpy as np
    root=Path(root);count=verify_manifest(root);spec=json.loads((root/'model-spec.json').read_text(encoding='utf8'));assert len(spec['layers'])==16
    keys=(root/'gpu/keys.txt').read_text(encoding='utf8').splitlines();assert len(keys)==18710 and keys[0]==''
    graph=(root/'gpu/cnn/tail.param').read_text();assert graph.count('NpuConv855 ')==16
    import onnx
    for layer in spec['layers']:
        name=layer['name'];ci=layer['ci'];co=layer['co'];folder=root/'npu'/name;bridge=np.fromfile(folder/'bridge.f32',dtype='<f4')
        assert len(bridge)==2*ci+3*co and np.isfinite(bridge).all() and (bridge[:ci]>0).all() and (bridge[2*ci:2*ci+co]>0).all(),name
        verify_csr(folder/'outliers.csr',co,ci,bridge[ci:2*ci]);onnx.checker.check_model(str(folder/'model.onnx'))
        weight=onnx.numpy_helper.to_array(onnx.load(folder/'model.onnx').graph.initializer[0]);assert weight.shape==(co,ci,1,1) and np.isfinite(weight).all(),name
        enc=json.loads((folder/'encodings.json').read_text())
        for value in [enc['weights'],*enc['activations'].values()]:
            assert value[4]==8 and value[2]>0 and -255<=value[3]<=0 and int(value[3])==value[3]
            assert abs(value[0]-value[2]*value[3])<value[2]*1e-3
        e=enc['weights'];q=np.rint(weight/e[2]-e[3]);assert q.min()>=0 and q.max()<=255 and np.max(abs(weight-(q+e[3])*e[2]))<e[2]*.001,name
    return count
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path);a=p.parse_args();print('ASSETS_VERIFIED',verify(a.assets))
