"""Run using SDK Python 3.6; no SDK headers or runtime are redistributed."""
from pathlib import Path
import argparse,json
import numpy as np,onnx
from onnx import numpy_helper
from qti.aisw.dlc_utils import modeltools
p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();folder=Path(a.folder);wanted=json.loads((folder/'encodings.json').read_text());m=modeltools.Model();m.load(str(folder/'int8-hta.dlc'));activations={};weight=None;weight_enc=None
for layer in m.get_layers():
    for i,name in enumerate(layer['output_names']):activations[name]=m.get_tf_output_encoding_by_index(layer['name'],i)
    if 'weights' in layer:weight=layer['weights'];weight_enc=m.get_tf_weight_encoding(layer['name'],0)
for name,enc in dict(wanted['activations'],weights=wanted['weights']).items():
    actual=weight_enc if name=='weights' else activations[name]
    assert actual[4]==enc[4] and actual[3]==enc[3] and abs(actual[2]-enc[2])/enc[2]<2e-6,(name,actual,enc)
expected=numpy_helper.to_array(onnx.load(str(folder/'model.onnx')).graph.initializer[0]).transpose(2,3,1,0);error=float(abs(expected-weight).max());assert error<wanted['weights'][2]*.01,error
print('DLC_AUDIT_PASS',folder.name,'max_weight_difference',error)
