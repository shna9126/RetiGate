.PHONY: setup test data exp01 exp02 exp03 exp04 exp05 exp06 all clean

PYTHON = .venv/bin/python
PIP    = .venv/bin/pip

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest tests/ -v

data:
	@echo ""
	@echo "=== DATASET DOWNLOAD INSTRUCTIONS ==="
	@echo ""
	@echo "1. KITTI Flow 2015 (required -- 1.6 GB):"
	@echo "   URL: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"
	@echo "   Unzip to: data/kitti/"
	@echo "   You need: training/image_2/, training/label_2/, training/flow_occ/"
	@echo "   Note: label_2/ is in the object detection package, not scene flow."
	@echo "   Also download: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
	@echo ""
	@echo "2. DAVIS 2017 480p (required -- 794 MB):"
	@echo "   URL: https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
	@echo "   Unzip to: data/davis/"
	@echo ""
	@echo "3. Middlebury Other (required -- ~45 MB):"
	@echo "   Images: https://vision.middlebury.edu/flow/data/comp/zip/other-color-allframes.zip"
	@echo "   GT Flow: https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip"
	@echo "   Unzip to: data/middlebury/"
	@echo ""
	@echo "4. KITTI-MoSeg (required for mAP evaluation -- moving object GT):"
	@echo "   Paper: Mohamed et al. 2020 'Monocular Instance Motion Segmentation'"
	@echo "   GitHub: https://github.com/Bariaw/KITTI-MoSeg"
	@echo "   Download the annotation files and place in: data/kitti/kitti_moseg/"
	@echo ""
	@echo "5. Synthetic data: auto-generated, no download needed."
	@echo ""
	@echo "After downloading, run:"
	@echo "  python -c \"from retigate.datasets.kitti import KITTIDataset; d = KITTIDataset(); print(f'KITTI: {len(d)} samples')\""
	@echo ""

exp01:
	$(PYTHON) experiments/01_cross_domain.py

exp02:
	$(PYTHON) experiments/02_baselines.py

exp03:
	$(PYTHON) experiments/03_ablation.py

exp04:
	$(PYTHON) experiments/04_detection_mAP.py

exp05:
	$(PYTHON) experiments/05_sahi_comparison.py

exp06:
	$(PYTHON) experiments/06_pareto_sweep.py

all: exp01 exp02 exp03 exp04 exp05 exp06

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
