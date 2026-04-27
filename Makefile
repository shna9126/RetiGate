# --- SETTINGS ---
.PHONY: setup test data fidelity map efficiency synthesis all clean

PYTHON = .venv/bin/python
PIP    = .venv/bin/pip

# --- INITIALIZATION ---
setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

clean:
	@echo "Cleaning up build artifacts and caches..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf *.egg-info .pytest_cache

# --- DATASET MANAGEMENT ---
data:
	@echo "=== DATASET PREPARATION (KITTI TRACKING & DAVIS) ==="
	@echo "1. KITTI Tracking (Required for mAP Audit):"
	@echo "   URL: https://www.cvlibs.net/datasets/kitti/eval_tracking.php"
	@echo "   Download: 'data_tracking_image_2.zip' and 'data_tracking_label_2.zip'"
	@echo "   Place in: data/kitti/data_tracking_image/image_02/ and .../label_02/"
	@echo ""
	@echo "2. DAVIS 2017 (Required for Generalization):"
	@echo "   Unzip to: data/davis/"

# --- SECTION 1: SENSING FIDELITY (RECALL) ---
fidelity:
	@echo "Running KITTI Tracking Fidelity Audit (Target: 95.5%+ Recall)..."
	$(PYTHON) experiments/02_Fidelity_Evaluations/20_kitti_tracking_audit.py
	@echo "Running DAVIS Pareto Sweep..."
	$(PYTHON) experiments/02_Fidelity_Evaluations/16_davis_pareto_sweep.py

# --- SECTION 2: SYSTEM ACCURACY (mAP) ---
map:
	@echo "Running Rigorous mAP Audit (COCO Protocol)..."
	$(PYTHON) experiments/02_Fidelity_Evaluations/14_kitti_map_audit.py

# --- SECTION 3: COMPUTATIONAL EFFICIENCY ---
efficiency:
	@echo "Profiling Power and FLOPs..."
	$(PYTHON) experiments/03_Efficiency_Evaluations/08_power_measurement.py
	@echo "Measuring Pipelined Throughput (FPS)..."
	$(PYTHON) experiments/03_Efficiency_Evaluations/18_async_pipeline.py
	@echo "Testing 4K Scalability..."
	$(PYTHON) experiments/03_Efficiency_Evaluations/19_scale_to_4k.py

# --- FINAL SYNTHESIS ---
synthesis:
	@echo "Generating Final Table of Results..."
	$(PYTHON) experiments/04_Synthesis_and_Viz/13_final_synthesis.py

all: fidelity map efficiency synthesis