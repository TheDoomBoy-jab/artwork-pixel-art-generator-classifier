"""Audit the two collections; matching stems are evidence candidates, not proof of pairing."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    collections = {}
    for name in ('sketches', 'pixel_art'):
        files = sorted(p for p in (root/name).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        collections[name] = [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    matching = {Path(p['file']).stem for p in collections['sketches']} & {Path(p['file']).stem for p in collections['pixel_art']}
    return {'collections': collections, 'counts': {k:len(v) for k,v in collections.items()}, 'matching_stems': sorted(matching), 'paired_training_verified': False, 'note': 'Explicit correspondence must be verified. Sorting independent collections does not create paired training data.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);args=p.parse_args()
    print(json.dumps(audit(args.data),indent=2))
