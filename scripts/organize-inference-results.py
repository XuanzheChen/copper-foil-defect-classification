#!/usr/bin/env python3
"""Copy inference inputs into Top-1 class folders and write one low-confidence JSON.

Runs locally with original images and the server's inference JSONL/summary.
Preserves relative filenames, checks complete coverage and SHA-256, and never moves
or overwrites original images. No model or training dependencies are imported.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe relative path: {value}')
    return path.as_posix()


def finite_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def organize(data, predictions, summary_path, manifest_path, labels_path, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    labels = []
    for line in Path(labels_path).read_text(encoding='utf-8-sig').splitlines():
        index, separator, label = line.partition(':')
        if separator != ':' or index != str(len(labels)) or relative_path(label) != label or '/' in label:
            raise ValueError('Invalid sequential class labels')
        labels.append(label)
    if len(labels) != 7 or len(set(labels)) != 7:
        raise ValueError('Expected seven unique class labels')
    if summary['k'] != 3 or summary['conf'] != 0.7:
        raise ValueError('Expected inference with k=3 and conf=0.7')
    files = {}
    for record in manifest['files']:
        name = relative_path(record['path'])
        if name in files:
            raise ValueError(f'Duplicate manifest path: {name}')
        source = data / name
        if source.is_symlink() or not source.is_file() or data not in source.resolve().parents:
            raise ValueError(f'Invalid source image: {name}')
        if source.stat().st_size != record['bytes'] or sha256(source) != record['sha256']:
            raise ValueError(f'Source differs from uploaded snapshot: {name}')
        files[name] = record
    actual = {p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
    if actual != set(files) or len(files) != manifest['total_files']:
        raise ValueError('Input file set differs from upload manifest')
    expected = {name for name, r in files.items() if r['is_image'] and (r['width'], r['height']) == (200, 200)}
    excluded = sum(r['is_image'] for r in files.values()) - len(expected)
    source_root = PurePosixPath(summary['data'])
    rows, seen = [], set()
    for line in Path(predictions).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        try:
            name = relative_path(PurePosixPath(row['path']).relative_to(source_root).as_posix())
        except ValueError:
            raise ValueError('Prediction path is outside recorded input root') from None
        if name not in expected or name in seen:
            raise ValueError(f'Unexpected or duplicate prediction: {name}')
        seen.add(name)
        conf, label = row['top1_conf'], row['pred_label']
        if label not in labels or not finite_probability(conf) or type(row['low_confidence']) is not bool:
            raise ValueError(f'Invalid Top-1 prediction: {name}')
        low = conf < 0.7
        if row['low_confidence'] != low or len(row['topk']) != (3 if low else 0):
            raise ValueError(f'Invalid strict threshold/Top-3 output: {name}')
        if low:
            candidates = row['topk']
            if (candidates[0]['class_name'], candidates[0]['confidence']) != (label, conf):
                raise ValueError('Top-3 first candidate differs from Top-1')
            ids = []
            last = 1.0
            for candidate in candidates:
                i, probability = candidate['class_id'], candidate['confidence']
                if type(i) is not int or not 0 <= i < 7 or candidate['class_name'] != labels[i]:
                    raise ValueError('Invalid Top-3 class')
                if not finite_probability(probability) or probability > last or i in ids:
                    raise ValueError('Invalid Top-3 probabilities/order')
                ids.append(i)
                last = probability
        rows.append((name, row))
    if seen != expected or len(rows) != summary['samples'] or len(expected) != manifest['admissible_images']:
        raise ValueError('Predictions do not cover all admissible images exactly once')
    if excluded != summary['excluded_non_200x200']:
        raise ValueError('Excluded image count differs from manifest')
    if sum(row['low_confidence'] for _, row in rows) != summary['low_confidence_samples']:
        raise ValueError('Low-confidence count differs from inference summary')
    names = ('by_top1', 'top1_predictions.csv', 'low_confidence_top3.json', 'organization_summary.json')
    if any((output / name).exists() for name in names):
        raise ValueError('Refusing to overwrite existing organized results')
    for label in labels:
        (output / 'by_top1' / label).mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(labels, 0)
    low_counts = dict.fromkeys(labels, 0)
    low_rows = []
    with (output / 'top1_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_path', 'organized_path', 'top1_label', 'top1_conf', 'low_confidence', 'sha256'])
        writer.writeheader()
        for name, row in rows:
            label = row['pred_label']
            relative = (PurePosixPath('by_top1') / label / name).as_posix()
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data / name, destination)
            if sha256(destination) != files[name]['sha256']:
                raise ValueError(f'Copied image hash mismatch: {name}')
            counts[label] += 1
            record = {'source_path': name, 'organized_path': relative, 'top1_label': label,
                      'top1_conf': row['top1_conf'], 'low_confidence': row['low_confidence'],
                      'sha256': files[name]['sha256']}
            writer.writerow(record)
            if row['low_confidence']:
                low_counts[label] += 1
                low_rows.append({**record, 'top3': row['topk']})
    document = {'schema_version': 1, 'source_root': str(source_root), 'conf': 0.7, 'k': 3,
                'comparison': 'top1_conf < conf (strict)', 'samples': len(low_rows), 'records': low_rows}
    (output / 'low_confidence_top3.json').write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    report = {'source_files': len(files), 'source_images': sum(r['is_image'] for r in files.values()),
              'classified_images': len(rows), 'excluded_non_200x200': excluded,
              'non_image_files': sum(not r['is_image'] for r in files.values()), 'per_class': counts,
              'low_confidence_per_class': low_counts, 'low_confidence_samples': len(low_rows),
              'conf': 0.7, 'k': 3, 'original_images_preserved': True, 'all_copied_image_hashes_verified': True,
              'manifest_sha256': sha256(manifest_path), 'predictions_sha256': sha256(predictions),
              'checkpoint_sha256': summary['checkpoint_sha256'], 'class_labels_sha256': summary['class_labels_sha256']}
    (output / 'organization_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'predictions', 'summary', 'manifest', 'class-labels', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    try:
        report = organize(args.data, args.predictions, args.summary, args.manifest, args.class_labels, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
