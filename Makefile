PY := .venv/bin/python

.PHONY: setup data train compress planning pipeline figures gnn test all
setup:
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -e ".[dev]"
data:        ## download the METU concrete-crack set (CC BY 4.0) into data/raw/ (git-ignored, untrusted data)
	$(PY) -m gnn_inspect.crack.download
train:       ## fine-tune ResNet-18 and DeiT-Tiny (MPS if available), writes results/crack/*_test.json
	$(PY) -m gnn_inspect.crack.train --arch resnet18
	$(PY) -m gnn_inspect.crack.train --arch deit_tiny
compress:    ## int8 quantization + magnitude pruning, accuracy vs size vs CPU latency
	$(PY) -m gnn_inspect.crack.compress --arch resnet18
	$(PY) -m gnn_inspect.crack.compress --arch deit_tiny
planning:    ## greedy+A* vs OR-Tools
	$(PY) -m gnn_inspect.bench_planning --sites 30
gnn:         ## original synthetic GNN-vs-centrality study
	$(PY) -m gnn_inspect.experiment --seeds 5
pipeline:    ## detector -> graph -> priorities -> route
	$(PY) -m gnn_inspect.pipeline --arch resnet18 --seeds 3
figures:
	$(PY) -m gnn_inspect.figures
test:
	$(PY) -m pytest -q
all: data train compress planning pipeline figures test
