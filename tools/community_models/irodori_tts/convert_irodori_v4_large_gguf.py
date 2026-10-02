#!/usr/bin/env python3
"""Package the original Aratako/Irodori-TTS-v4-Large checkpoint for audio.cpp.

Only the safetensors header is read in Python. Native audiocpp_gguf converts
weights and embeds the checkpoint's configs and bundled tokenizer.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


def read_header(path):
    with path.open('rb') as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError('truncated safetensors header')
        length, = struct.unpack('<Q', length_bytes)
        if length > 64 * 1024 * 1024 or length > path.stat().st_size - 8:
            raise ValueError('invalid safetensors header length')
        return json.loads(f.read(length))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True, help='Downloaded original Large repository')
    p.add_argument('--codec', type=Path, required=True, help='Semantic-DACVAE weights.safetensors')
    p.add_argument('--converter', type=Path, required=True, help='Built audiocpp_gguf executable')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--type', choices=['f16', 'q8_0'], default='f16')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    source = args.source_dir.resolve()
    checkpoint = source / 'model.safetensors'
    for path in [checkpoint, args.codec, args.converter, source/'tokenizer/tokenizer.json',
                 source/'tokenizer/tokenizer_config.json']:
        if not path.is_file():
            p.error(f'missing file: {path}')
    if args.output.exists() and not args.overwrite:
        p.error(f'output exists: {args.output} (use --overwrite to replace it)')
    header = read_header(checkpoint)
    metadata = header.get('__metadata__', {})
    config = json.loads(metadata['config_json'])
    encoder = json.loads(metadata['text_encoder_config_json'])
    if encoder.get('model_type') != 't5gemma2' or config.get('text_encoder_type') != 'pretrained':
        p.error('expected the original Irodori v4 Large T5Gemma2 checkpoint')
    embedding = header.get('pretrained_text_backbone.backbone.embed_tokens.weight', {})
    if embedding.get('shape') != [config['text_vocab_size'], encoder['encoder']['text_config']['hidden_size']]:
        p.error('missing or incompatible text embedding; packed quantized checkpoints are not supported')
    if any(name.endswith(('._weight_qdata', '._weight_scale')) for name in header):
        p.error('packed quantized checkpoints are not supported')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='irodori-large-') as temp:
        root = Path(temp)
        (root/'model_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        (root/'text_encoder_config.json').write_text(json.dumps(encoder, ensure_ascii=False, indent=2), encoding='utf-8')
        (root/'tokenizer').mkdir()
        for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
            if (source/'tokenizer'/name).is_file():
                shutil.copy2(source/'tokenizer'/name, root/'tokenizer'/name)
        # Preserve source licensing and notices in the self-contained package.
        notices = []
        if (source/'README.md').is_file():
            notices += ['--sidecar', f'{source / "README.md"}=UPSTREAM_README.md']
        for pattern in ['LICENSE*', 'NOTICE*']:
            for path in source.glob(pattern):
                if path.is_file():
                    notices += ['--sidecar', f'{path}={path.name}']
        command = [str(args.converter.resolve()), '--input', f'model_weights={checkpoint}',
                   '--input', f'codec_weights={args.codec.resolve()}', '--root', str(root),
                   '--family', 'irodori_tts', '--type', args.type, '--output', str(args.output.resolve())]
        if args.overwrite:
            command.append('--overwrite')
        subprocess.run(command + notices, check=True)


if __name__ == '__main__':
    main()
