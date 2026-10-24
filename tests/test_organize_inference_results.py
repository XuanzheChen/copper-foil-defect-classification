"""Coverage, confidence boundaries, collision handling, and snapshot integrity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('organize_results', Path(__file__).resolve().parents[1] / 'scripts/organize-inference-results.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class OrganizeInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / 'input'
        self.output = self.root / 'results'
        self.labels = ['凹凸', '刮伤', '压坑', '异物', '拖尾', '铜瘤', '黑点']
        self.labels_file = self.root / 'labels.txt'
        self.labels_file.write_text('\n'.join(f'{i}:{name}' for i, name in enumerate(self.labels)), encoding='utf-8')
        files = []
        for name, size in [('a/same.png', (200, 200)), ('b/same.png', (200, 200)), ('large.png', (200, 240)), ('meta.xml', None)]:
            target = self.data / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = name.encode()
            target.write_bytes(content)
            row = dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), is_image=size is not None)
            if size:
                row.update(width=size[0], height=size[1])
            files.append(row)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(files=files, total_files=4, admissible_images=2)), encoding='utf-8')
        self.summary = self.root / 'summary.json'
        self.summary.write_text(json.dumps(dict(data='/server/input', k=3, conf=0.7, samples=2, excluded_non_200x200=1,
                                               low_confidence_samples=1, checkpoint_sha256='weight', class_labels_sha256='labels')), encoding='utf-8')
        self.rows = [dict(path='/server/input/a/same.png', pred_label='凹凸', top1_conf=0.7, low_confidence=False, topk=[]),
                     dict(path='/server/input/b/same.png', pred_label='凹凸', top1_conf=0.6999999, low_confidence=True,
                          topk=[dict(class_id=i, class_name=self.labels[i], confidence=c) for i, c in [(0, 0.6999999), (2, 0.2), (5, 0.1)]])]
        self.predictions = self.root / 'predictions.jsonl'
        self.write_predictions()

    def write_predictions(self):
        self.predictions.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in self.rows), encoding='utf-8')

    def run_organizer(self):
        return module.organize(self.data, self.predictions, self.summary, self.manifest, self.labels_file, self.output)

    def test_copies_preserve_nested_duplicate_names_and_strict_threshold(self):
        result = self.run_organizer()
        self.assertEqual(result['classified_images'], 2)
        self.assertEqual(result['excluded_non_200x200'], 1)
        self.assertEqual(len(list((self.output / 'by_top1').iterdir())), 7)
        for folder in ('a', 'b'):
            self.assertEqual((self.output / 'by_top1/凹凸' / folder / 'same.png').read_bytes(), (self.data / folder / 'same.png').read_bytes())
        low = json.loads((self.output / 'low_confidence_top3.json').read_text(encoding='utf-8'))
        self.assertEqual(low['samples'], 1)
        self.assertEqual(low['records'][0]['source_path'], 'b/same.png')
        self.assertEqual([r['class_name'] for r in low['records'][0]['top3']], ['凹凸', '压坑', '铜瘤'])
        self.assertTrue((self.data / 'large.png').exists())

    def test_missing_or_duplicate_predictions_fail_before_copying(self):
        for rows in (self.rows[:1], self.rows + self.rows[:1]):
            self.predictions.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.run_organizer()
            self.assertFalse(self.output.exists())

    def test_tampered_source_rejected_before_copying(self):
        (self.data / 'a/same.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            self.run_organizer()
        self.assertFalse(self.output.exists())

    def test_outside_source_prediction_rejected(self):
        self.rows[0]['path'] = '/server/other/a.png'
        self.write_predictions()
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.run_organizer()

    def test_wrong_top3_mapping_rejected(self):
        self.rows[1]['topk'][1]['class_name'] = '黑点'
        self.write_predictions()
        with self.assertRaisesRegex(ValueError, 'Top-3 class'):
            self.run_organizer()
        self.assertFalse(self.output.exists())

    def test_existing_results_are_preserved(self):
        self.run_organizer()
        before = (self.output / 'low_confidence_top3.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_organizer()
        self.assertEqual((self.output / 'low_confidence_top3.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
