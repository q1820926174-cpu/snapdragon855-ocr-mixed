"""Verify the separately downloadable, SDK-free PNNX GPU model package."""
from pathlib import Path
import argparse
import json
from collections import Counter
from verify_models import verify_manifest


def verify(root):
    root = Path(root)
    count = verify_manifest(root)
    manifest = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf8'))
    required = {'keys.txt', 'export.json', 'model-spec.json', 'source/original.onnx'}
    required.update(f'{part}/model.{kind}' for part in ('full', 'cnn', 'tail') for kind in ('param', 'bin'))
    if not required.issubset(manifest):
        raise ValueError('Incomplete GPU model manifest')
    spec = json.loads((root / 'model-spec.json').read_text(encoding='utf8'))
    if spec['input_nhwc'] != [1, 48, 640, 3] or spec['ctc_classes'] != 18710:
        raise ValueError('Unsupported model contract')
    keys = (root / 'keys.txt').read_text(encoding='utf8').splitlines()
    if len(keys) != 18710 or keys[0] != '':
        raise ValueError('Invalid CTC dictionary')
    for part in ('full', 'cnn', 'tail'):
        graph = (root / part / 'model.param').read_text(encoding='utf8')
        if 'NpuConv855' in graph or 'Squeeze ' in graph:
            raise ValueError('Unexpected lowered or unsupported graph')
        operators = Counter(line.split()[0] for line in graph.splitlines()[2:])
        expected = {'full': (14, 5, 2), 'cnn': (14, 0, 0), 'tail': (0, 5, 2)}[part]
        if tuple(operators[name] for name in ('GELU', 'LayerNorm', 'MultiHeadAttention')) != expected:
            raise ValueError('Native operator contract changed')
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('assets', type=Path)
    args = parser.parse_args()
    print('GPU_ASSETS_VERIFIED', verify(args.assets))
