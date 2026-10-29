"""Export native GPU graphs directly from the original Paddle ONNX model.

No SNPE lowering, training, or output-label adjustment. The standalone PNNX
binary is located without importing its optional Python/Torch frontend.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,importlib.metadata,platform,shutil
from collections import Counter
import onnx

EXPECTED_SOURCE='f12a745afc282284d630cd8c60ab4f503945fa830a13c8fa8d8b90c588349961'

def export(source,output):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash!=EXPECTED_SOURCE:raise ValueError('Use the original reference source supplied with the GPU model asset')
    dist=importlib.metadata.distribution('pnnx');binary=Path(dist.locate_file('pnnx'))/('pnnx.exe' if platform.system()=='Windows' else 'pnnx')
    if not binary.exists():raise FileNotFoundError(binary)
    original=onnx.load(source);weights={t.name:hashlib.sha256(t.SerializeToString()).hexdigest() for t in original.graph.initializer};records=[]
    for part in ['full','cnn','tail']:
        model=onnx.load(source)
        for d,value in zip(model.graph.input[0].type.tensor_type.shape.dim,[1,3,48,640]):d.ClearField('dim_param');d.dim_value=value
        model=onnx.shape_inference.infer_shapes(model);last=model.graph.node[-1];assert last.op_type=='Softmax';logit=last.input[0];feature=next(n.output[0] for n in model.graph.node if n.op_type=='AveragePool')
        before=feature if part=='tail' else model.graph.input[0].name;after=feature if part=='cnn' else logit;shape=[1,768,1,80] if part=='tail' else [1,3,48,640];newinput='t_293' if part=='tail' else 't_0';newoutput='t_293' if part=='cnn' else 't_469';model=onnx.utils.Extractor(model).extract_model([before],[after])
        for node in model.graph.node:
            for names in [node.input,node.output]:
                for i,name in enumerate(names):
                    if name==before:names[i]=newinput
                    elif name==after:names[i]=newoutput
        for value in [*model.graph.input,*model.graph.output,*model.graph.value_info]:
            if value.name==before:value.name=newinput
            elif value.name==after:value.name=newoutput
        for d,value in zip(model.graph.input[0].type.tensor_type.shape.dim,shape):d.ClearField('dim_param');d.dim_value=value
        assert all(weights[t.name]==hashlib.sha256(t.SerializeToString()).hexdigest() for t in model.graph.initializer)
        onnx.checker.check_model(model);name='original-logits' if part=='full' else 'original-'+part;onnx.save(model,output/(name+'.onnx'))
        with (output/(part+'-convert.log')).open('wb') as log:subprocess.run([str(binary),name+'.onnx','inputshape=['+','.join(map(str,shape))+']','fp16=0'],cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True)
        stem=name.replace('-','_');lines=(output/(stem+'.ncnn.param')).read_text().splitlines();repairs=[]
        for i,line in enumerate(lines[2:],2):
            tokens=[newinput if token=='in0' else newoutput if token=='out0' else token for token in line.split()]
            if tokens[0]=='Squeeze':
                assert part in ['full','tail'] and tokens[6:]==['-23303=1,1']
                repairs.append(dict(layer=tokens[1],source_shape=[1,192,1,80],target_shape=[1,192,80],replacement='Exact static Reshape, ncnn shape 80x192'));tokens=['Reshape',*tokens[1:6],'0=80','1=192']
            lines[i]=' '.join(tokens)
        target=output/part;target.mkdir(exist_ok=True);(target/'model.param').write_bytes(('\n'.join(lines)+'\n').encode('utf8'));shutil.copyfile(output/(stem+'.ncnn.bin'),target/'model.bin');records.append(dict(part=part,source_weights_unchanged_before_pnnx=True,operators=dict(Counter(line.split()[0] for line in lines[2:])),layout_repairs=repairs));print('GPU_PNNX_EXPORTED',part,flush=True)
    (output/'export.json').write_text(json.dumps(dict(source_sha256=source_hash,pnnx_version=dist.version,original_erf_preserved=True,offline_weight_storage='FP32; PNNX may fold/fuse weights',records=records),indent=2))
    return records

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();export(args.source,args.output)
