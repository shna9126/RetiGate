# RetiGate

RetiGate is a bio-inspired motion-saliency pre-filter for object detection, modelled on the retinal circuitry of direction-selective ganglion cells. It suppresses static background content and focuses downstream detectors on kinetic regions, reducing compute and improving detection efficiency on moving objects.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Or with Make:

```bash
make setup
```

## Dataset Download

```bash
make data
```

This prints instructions for downloading KITTI Flow 2015, DAVIS 2017, Middlebury Other, and KITTI-MoSeg. Datasets are not auto-downloaded due to license requirements.

## Running Experiments

```bash
make all        # runs experiments 01–06 in sequence
make exp01      # 01_cross_domain.py only
make exp02      # 02_baselines.py only
# ... etc.
```

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

## Hardware

Tested on Apple M3 Pro, macOS 15 (Sequoia). All timing benchmarks were measured on this hardware.
