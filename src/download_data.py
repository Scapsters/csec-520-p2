"""Acquire the course UNSW-NB15 training CSV, verifying its exact bytes."""
import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

URL = 'https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv'
SHA256 = 'bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa'


def checksum(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(path='data/UNSW_NB15_training-set.csv', url=URL):
    path = Path(path)
    if path.exists():
        if checksum(path) != SHA256:
            raise ValueError(f'{path} has a different checksum; it was not overwritten')
        print('Dataset verified:', path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
            staged = Path(target.name)
            with urlopen(url, timeout=120) as response:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    target.write(block)
        if checksum(staged) != SHA256:
            raise ValueError('Downloaded dataset checksum mismatch; destination was not written')
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    print('Dataset downloaded and verified:', path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/UNSW_NB15_training-set.csv')
    parser.add_argument('--url', default=URL, help='Alternate URL serving identical training CSV bytes')
    args = parser.parse_args()
    download(args.path, args.url)
