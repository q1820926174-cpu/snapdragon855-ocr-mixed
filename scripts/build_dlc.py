"""Compile model-owned tensors with a user-provided licensed SNPE 1.59 SDK."""
from pathlib import Path
import argparse,os,json,subprocess,shutil
from verify_models import verify_manifest
p=argparse.ArgumentParser();p.add_argument('--sdk',required=True,type=Path);p.add_argument('--sdk-python',required=True,type=Path);p.add_argument('--assets',type=Path,default=Path(__file__).resolve().parents[1]/'models');p.add_argument('--work',required=True,type=Path);a=p.parse_args();sdk=a.sdk.resolve();work=a.work.resolve();work.mkdir(parents=True,exist_ok=True);assert str(work).isascii(),'Legacy SDK work directory must use ASCII';verify_manifest(a.assets)
env=os.environ.copy();env['PYTHONPATH']=str(sdk/'lib/python');env['LD_LIBRARY_PATH']=str(a.sdk_python.resolve().parent.parent/'lib')+':'+str(sdk/'lib/x86_64-linux-clang')+':'+env.get('LD_LIBRARY_PATH','');env['OMP_NUM_THREADS']='2';env['OMP_WAIT_POLICY']='PASSIVE'
def encoding(value):return dict(bitwidth=int(value[4]),min=value[0],max=value[1],scale=value[2],offset=int(value[3]),is_symmetric='False')
def run(command,log):
    with log.open('wb') as output:subprocess.run([str(v) for v in command],env=env,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=180)
spec=json.loads((a.assets/'model-spec.json').read_text());records=[]
for layer in spec['layers']:
    source=a.assets/'npu'/layer['name'];folder=work/layer['name'];folder.mkdir(exist_ok=True)
    for name in ['model.onnx','encodings.json']:shutil.copy2(source/name,folder/name)
    enc=json.loads((source/'encodings.json').read_text());external=dict(activation_encodings={k:[encoding(v)] for k,v in enc['activations'].items()},param_encodings={'weight':[encoding(enc['weights'])]});(folder/'overrides.json').write_text(json.dumps(external));(folder/'placeholder.raw').write_bytes(bytes(layer['ci']*layer['h']*layer['w']*4));(folder/'inputs.txt').write_text('x:='+str(folder/'placeholder.raw')+'\n')
    run([a.sdk_python,sdk/'bin/x86_64-linux-clang/snpe-onnx-to-dlc','-i',folder/'model.onnx','-o',folder/'float.dlc','--quantization_overrides',folder/'overrides.json'],folder/'convert.log')
    run([sdk/'bin/x86_64-linux-clang/snpe-dlc-quantize','--input_dlc',folder/'float.dlc','--input_list',folder/'inputs.txt','--output_dlc',folder/'int8-hta.dlc','--override_params','--bias_bitwidth','32','--enable_hta','--hta_partitions','sm8150:0-1'],folder/'quantize.log')
    run([a.sdk_python,sdk/'bin/x86_64-linux-clang/snpe-dlc-info','-i',folder/'int8-hta.dlc'],folder/'compiled-info.txt');info=(folder/'compiled-info.txt').read_text();assert 'target: HTA' in info and 'PERMUTE' not in info.upper()
    run([a.sdk_python,Path(__file__).with_name('audit_dlc.py'),folder],folder/'audit.log')
    for name in ['int8-hta.dlc','compiled-info.txt','audit.log']:shutil.copy2(folder/name,source/name)
    records.append(layer['name']);print('HTA_MODEL_AUDITED',layer['name'],flush=True)
(work/'build-result.json').write_text(json.dumps(dict(passed=True,layers=records,weights_and_encodings_verified=True),indent=2))
