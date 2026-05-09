# Makefile de ejecucion experimental para hope_ff.
# Uso rapido:
#   make
#   make run TRAIN_SAMPLES=16 EPOCHS=1 HIDDEN_LAYERS=2 HIDDEN_VALUES="32 8" PARALLEL_NEURONS=8
#   make run IMG_W=16 IMG_H=16 PIX_BITS=1 DATASET=../mnist/data/processed/mnist_16x16_1b_packed.bin
#   make run-bins
#   make run-archs
#   make sweep

.DEFAULT_GOAL := run
MAKEFLAGS += --no-print-directory

CXX ?= g++
PYTHON ?= python
BASE_CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label -Iinclude
CXXFLAGS ?= $(BASE_CXXFLAGS)
THIS_DIR := $(subst \,/,$(CURDIR))
VITIS_ROOT ?= C:/Xilinx/Vitis/2024.2
VITIS_RUN ?= $(VITIS_ROOT)/bin/vitis-run.bat
HLS_PART ?= xc7a100tcsg324-1
HLS_WORK_ROOT ?= hls_work

BUILD_DIR ?= build
RUN_DIR ?= runs
RESULTS_DIR ?= results
DATA_DIR ?= ../mnist/data/processed
DATASETS := $(wildcard $(DATA_DIR)/*.bin)
GROUPS ?= smoke debug comparacion barrido_final
RUN_ALL_EXTRA ?=
RUN_TAG ?=
SWEEP100_RESULTS_DIR ?= results/sweep100
SWEEP100_LIMIT ?=
SWEEP100_EXTRA ?=

# Dataset/build seleccionados. Deben coincidir con la cabecera del .bin.
IMG_W ?= 28
IMG_H ?= 28
PIX_BITS ?= 1
DATASET ?= $(DATA_DIR)/mnist_$(IMG_W)x$(IMG_H)_$(PIX_BITS)b_packed.bin
DATASET_STEM := $(basename $(notdir $(DATASET)))
DATASET_FIELDS := $(subst _, ,$(DATASET_STEM))
PARSED_RES := $(word 2,$(DATASET_FIELDS))
PARSED_BITS_TOKEN := $(word 3,$(DATASET_FIELDS))
PARSED_IMG_W := $(word 1,$(subst x, ,$(PARSED_RES)))
PARSED_IMG_H := $(word 2,$(subst x, ,$(PARSED_RES)))
PARSED_PIX_BITS := $(patsubst %b,%,$(PARSED_BITS_TOKEN))

ifeq ($(origin DATASET), command line)
ifneq ($(origin IMG_W), command line)
IMG_W := $(PARSED_IMG_W)
endif
ifneq ($(origin IMG_H), command line)
IMG_H := $(PARSED_IMG_H)
endif
ifneq ($(origin PIX_BITS), command line)
PIX_BITS := $(PARSED_PIX_BITS)
endif
endif

# Hiperparametros de arquitectura y entrenamiento.
HIDDEN_LAYERS ?= 1
HIDDEN_VALUES ?= 64
PARALLEL_NEURONS ?= 8
THRESHOLD ?= 72
LABEL_SCALE ?= 8
PIXEL_SCALE ?= 1
PIXEL_LR ?= 1
LABEL_LR ?= 4
BIAS_LR ?= 1
ACT_CLIP ?= 255
LATENT_CLIP ?= 127
BIAS_CLIP ?= 2047

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
HIDDEN_VALUE_COUNT := $(words $(HIDDEN_VALUES))
ifneq ($(HIDDEN_LAYERS),$(HIDDEN_VALUE_COUNT))
$(error HIDDEN_LAYERS=$(HIDDEN_LAYERS) no coincide con HIDDEN_VALUES="$(HIDDEN_VALUES)" ($(HIDDEN_VALUE_COUNT) valores))
endif
HIDDEN_0 := $(word 1,$(HIDDEN_VALUES))
HIDDEN_1 := $(if $(word 2,$(HIDDEN_VALUES)),$(word 2,$(HIDDEN_VALUES)),1)
HIDDEN_2 := $(if $(word 3,$(HIDDEN_VALUES)),$(word 3,$(HIDDEN_VALUES)),1)
HIDDEN_3 := $(if $(word 4,$(HIDDEN_VALUES)),$(word 4,$(HIDDEN_VALUES)),1)
ARCH_TAG := L$(HIDDEN_LAYERS)_$(subst $(SPACE),_,$(strip $(HIDDEN_VALUES)))_P$(PARALLEL_NEURONS)

GROUP ?= custom
ifeq ($(GROUP),smoke)
GROUP_TRAIN_SAMPLES := 256
GROUP_EVAL_SAMPLES := 64
GROUP_EPOCHS := 1
GROUP_EVAL_EVERY_EPOCH := 0
else ifeq ($(GROUP),debug)
GROUP_TRAIN_SAMPLES := 4096
GROUP_EVAL_SAMPLES := 1024
GROUP_EPOCHS := 5
GROUP_EVAL_EVERY_EPOCH := 0
else ifeq ($(GROUP),comparacion)
GROUP_TRAIN_SAMPLES := 35700
GROUP_EVAL_SAMPLES := 6300
GROUP_EPOCHS := 20
GROUP_EVAL_EVERY_EPOCH := 1
else ifeq ($(GROUP),barrido_final)
GROUP_TRAIN_SAMPLES := 35700
GROUP_EVAL_SAMPLES := 6300
GROUP_EPOCHS := 30
GROUP_EVAL_EVERY_EPOCH := 1
else
GROUP_TRAIN_SAMPLES := 1024
GROUP_EVAL_SAMPLES := 512
GROUP_EPOCHS := 5
GROUP_EVAL_EVERY_EPOCH := 1
endif

TRAIN_SAMPLES ?= $(GROUP_TRAIN_SAMPLES)
EVAL_SAMPLES ?= $(GROUP_EVAL_SAMPLES)
EPOCHS ?= $(GROUP_EPOCHS)
EVAL_EVERY_EPOCH ?= $(GROUP_EVAL_EVERY_EPOCH)
SEED ?= 0x1234

ifndef TIMESTAMP
TIMESTAMP := $(shell $(PYTHON) -c "from datetime import datetime; n=datetime.now(); print(f'{n.year:04d}{n.month:02d}{n.day:02d}_{n.hour:02d}{n.minute:02d}{n.second:02d}')")
endif
VARIANT ?= $(IMG_W)x$(IMG_H)_$(PIX_BITS)b_$(ARCH_TAG)_thr$(THRESHOLD)_ls$(LABEL_SCALE)_ps$(PIXEL_SCALE)_plr$(PIXEL_LR)_llr$(LABEL_LR)
TARGET_DIR := $(BUILD_DIR)/$(VARIANT)
TARGET := $(TARGET_DIR)/train_tb.exe
HLS_CFG := $(TARGET_DIR)/hls_config.cfg
HLS_WORK_DIR := $(HLS_WORK_ROOT)/$(VARIANT)
RUN_NAME ?= $(VARIANT)_train$(TRAIN_SAMPLES)_eval$(EVAL_SAMPLES)_ep$(EPOCHS)_$(TIMESTAMP)
RESULT_DIR := $(RESULTS_DIR)/$(DATASET_STEM)/$(GROUP)/$(ARCH_TAG)$(if $(RUN_TAG),/$(RUN_TAG),)
LOG := $(RESULT_DIR)/log.txt
TMP_LOG := $(RUN_DIR)/$(RUN_NAME).program.tmp
HLS_CSIM_LOG := $(RUN_DIR)/$(RUN_NAME)_csim.txt
HLS_CSIM_TMP_LOG := $(RUN_DIR)/$(RUN_NAME)_csim.program.tmp

DEFINES := \
	-DFF_IMAGE_WIDTH=$(IMG_W) \
	-DFF_IMAGE_HEIGHT=$(IMG_H) \
	-DFF_PIXEL_BITS=$(PIX_BITS) \
	-DFF_HIDDEN_LAYERS=$(HIDDEN_LAYERS) \
	-DFF_HIDDEN_0=$(HIDDEN_0) \
	-DFF_HIDDEN_1=$(HIDDEN_1) \
	-DFF_HIDDEN_2=$(HIDDEN_2) \
	-DFF_HIDDEN_3=$(HIDDEN_3) \
	-DFF_PARALLEL_NEURONS=$(PARALLEL_NEURONS) \
	-DFF_GOODNESS_THRESHOLD=$(THRESHOLD) \
	-DFF_LABEL_SCALE=$(LABEL_SCALE) \
	-DFF_PIXEL_SCALE=$(PIXEL_SCALE) \
	-DFF_PIXEL_LR_STEP=$(PIXEL_LR) \
	-DFF_LABEL_LR_STEP=$(LABEL_LR) \
	-DFF_BIAS_LR_STEP=$(BIAS_LR) \
	-DFF_ACTIVATION_CLIP=$(ACT_CLIP) \
	-DFF_LATENT_CLIP=$(LATENT_CLIP) \
	-DFF_BIAS_CLIP=$(BIAS_CLIP)

HLS_DEFINES := -DFF_IMAGE_WIDTH=$(IMG_W) -DFF_IMAGE_HEIGHT=$(IMG_H) -DFF_PIXEL_BITS=$(PIX_BITS) -DFF_HIDDEN_LAYERS=$(HIDDEN_LAYERS) -DFF_HIDDEN_0=$(HIDDEN_0) -DFF_HIDDEN_1=$(HIDDEN_1) -DFF_HIDDEN_2=$(HIDDEN_2) -DFF_HIDDEN_3=$(HIDDEN_3) -DFF_PARALLEL_NEURONS=$(PARALLEL_NEURONS) -DFF_GOODNESS_THRESHOLD=$(THRESHOLD) -DFF_LABEL_SCALE=$(LABEL_SCALE) -DFF_PIXEL_SCALE=$(PIXEL_SCALE) -DFF_PIXEL_LR_STEP=$(PIXEL_LR) -DFF_LABEL_LR_STEP=$(LABEL_LR) -DFF_BIAS_LR_STEP=$(BIAS_LR) -DFF_ACTIVATION_CLIP=$(ACT_CLIP) -DFF_LATENT_CLIP=$(LATENT_CLIP) -DFF_BIAS_CLIP=$(BIAS_CLIP)
HLS_CFLAGS := -std=c++14 -I$(THIS_DIR)/include $(HLS_DEFINES)

SOURCES := \
	src/ff_dataset.cpp \
	src/ff_layer.cpp \
	src/ff_train.cpp \
	src/ff_infer.cpp \
	testbench/train_tb.cpp

.PHONY: help all build config run run-all summarize summarize-sweep100 sweep100 sweep100-dry list-datasets hls-config csim-run run-bins run-archs sweep clean clean-runs clean-hls clean-results \
	run-smoke run-debug run-comparacion run-barrido-final \
	run-bin-16x16-1b run-bin-20x20-1b run-bin-28x28-1b run-bin-20x20-4b run-bin-20x20-6b \
	run-arch-h32-p4 run-arch-h64-p8 run-arch-h96-p8 run-arch-h128-p8

help:
	@echo.
	@echo hope_ff Makefile
	@echo   make                 Compila, ejecuta y guarda results/.../log.txt
	@echo   make build           Solo compila la variante actual
	@echo   make run             Ejecuta la variante actual y guarda results/
	@echo   make run-all         Ejecuta DATASETS x GROUPS y guarda results/
	@echo   make summarize       Genera results/summary_all.md
	@echo   make sweep100        Ejecuta >=100 simulaciones barrido_final
	@echo   make summarize-sweep100 Genera tabla ordenada por accuracy/complejidad
	@echo   make list-datasets   Lista los .bin detectados
	@echo   make csim-run        Ejecuta Vitis HLS C-sim 2024.2 para Nexys A7
	@echo   make config          Muestra la configuracion efectiva
	@echo   make run-bins        Prueba varios .bin con la arquitectura actual
	@echo   make run-archs       Prueba varias arquitecturas sobre el .bin actual
	@echo   make sweep           Barrido corto de .bin + arquitecturas
	@echo   make clean           Borra build/
	@echo   make clean-runs      Borra runs/
	@echo   make clean-results   Borra results/
	@echo   make clean-hls       Borra hls_work/
	@echo.
	@echo Ejemplo:
	@echo   make run HIDDEN_LAYERS=2 HIDDEN_VALUES="32 8" PARALLEL_NEURONS=8 TRAIN_SAMPLES=16 EPOCHS=1
	@echo   make run DATASET=../mnist/data/processed/mnist_28x28_1b_packed.bin GROUP=smoke
	@echo   make run-all
	@echo   make summarize
	@echo   make csim-run HLS_PART=xc7a100tcsg324-1 TRAIN_SAMPLES=16 EPOCHS=1

all: build

build: $(TARGET)

config:
	@echo VARIANT=$(VARIANT)
	@echo TARGET=$(TARGET)
	@echo LOG=$(LOG)
	@echo HLS_CSIM_LOG=$(HLS_CSIM_LOG)
	@echo DATASET=$(DATASET)
	@echo DATASET_STEM=$(DATASET_STEM)
	@echo RESULTS_DIR=$(RESULTS_DIR)
	@echo RESULT_DIR=$(RESULT_DIR)
	@echo RUN_TAG=$(RUN_TAG)
	@echo DATASETS=$(DATASETS)
	@echo GROUP=$(GROUP)
	@echo GROUPS=$(GROUPS)
	@echo VITIS_RUN=$(VITIS_RUN)
	@echo THIS_DIR=$(THIS_DIR)
	@echo HLS_PART=$(HLS_PART)
	@echo HLS_CFG=$(HLS_CFG)
	@echo HLS_WORK_DIR=$(HLS_WORK_DIR)
	@echo IMG_W=$(IMG_W)
	@echo IMG_H=$(IMG_H)
	@echo PIX_BITS=$(PIX_BITS)
	@echo HIDDEN_LAYERS=$(HIDDEN_LAYERS)
	@echo HIDDEN_VALUES=$(HIDDEN_VALUES)
	@echo HIDDEN_0=$(HIDDEN_0)
	@echo HIDDEN_1=$(HIDDEN_1)
	@echo HIDDEN_2=$(HIDDEN_2)
	@echo HIDDEN_3=$(HIDDEN_3)
	@echo PARALLEL_NEURONS=$(PARALLEL_NEURONS)
	@echo ARCH_TAG=$(ARCH_TAG)
	@echo THRESHOLD=$(THRESHOLD)
	@echo LABEL_SCALE=$(LABEL_SCALE)
	@echo PIXEL_SCALE=$(PIXEL_SCALE)
	@echo PIXEL_LR=$(PIXEL_LR)
	@echo LABEL_LR=$(LABEL_LR)
	@echo BIAS_LR=$(BIAS_LR)
	@echo TRAIN_SAMPLES=$(TRAIN_SAMPLES)
	@echo EVAL_SAMPLES=$(EVAL_SAMPLES)
	@echo EPOCHS=$(EPOCHS)
	@echo EVAL_EVERY_EPOCH=$(EVAL_EVERY_EPOCH)
	@echo SEED=$(SEED)
	@echo SWEEP100_RESULTS_DIR=$(SWEEP100_RESULTS_DIR)
	@echo SWEEP100_LIMIT=$(SWEEP100_LIMIT)
	@echo SWEEP100_EXTRA=$(SWEEP100_EXTRA)

$(TARGET_DIR):
	@$(PYTHON) -c "from pathlib import Path; Path(r'$(TARGET_DIR)').mkdir(parents=True, exist_ok=True)"

$(RUN_DIR):
	@$(PYTHON) -c "from pathlib import Path; Path(r'$(RUN_DIR)').mkdir(parents=True, exist_ok=True)"

$(TARGET): $(SOURCES) | $(TARGET_DIR)
	@$(CXX) $(CXXFLAGS) $(DEFINES) $(SOURCES) -o $(TARGET)

run: $(TARGET) | $(RUN_DIR)
	@$(PYTHON) scripts/run_one_experiment.py --exe "$(TARGET)" --dataset "$(DATASET)" --train-samples "$(TRAIN_SAMPLES)" --eval-samples "$(EVAL_SAMPLES)" --epochs "$(EPOCHS)" --eval-every-epoch "$(EVAL_EVERY_EPOCH)" --run-name "$(RUN_NAME)" --group "$(GROUP)" --output-dir "$(RESULT_DIR)" --hidden-layers "$(HIDDEN_LAYERS)" --hidden-values "$(HIDDEN_VALUES)" --parallel-neurons "$(PARALLEL_NEURONS)" --seed "$(SEED)"

list-datasets:
	@$(PYTHON) -c "from pathlib import Path; [print(p.as_posix()) for p in sorted(Path('$(DATA_DIR)').glob('*.bin'))]"

run-all:
	@$(PYTHON) scripts/run_all_experiments.py --make "$(MAKE)" --data-dir "$(DATA_DIR)" --results-dir "$(RESULTS_DIR)" --groups "$(GROUPS)" -- "HIDDEN_LAYERS=$(HIDDEN_LAYERS)" "HIDDEN_VALUES=$(HIDDEN_VALUES)" "PARALLEL_NEURONS=$(PARALLEL_NEURONS)" $(RUN_ALL_EXTRA)

run-smoke:
	@$(MAKE) run-all GROUPS=smoke RUN_ALL_EXTRA="$(RUN_ALL_EXTRA)"

run-debug:
	@$(MAKE) run-all GROUPS=debug RUN_ALL_EXTRA="$(RUN_ALL_EXTRA)"

run-comparacion:
	@$(MAKE) run-all GROUPS=comparacion RUN_ALL_EXTRA="$(RUN_ALL_EXTRA)"

run-barrido-final:
	@$(MAKE) run-all GROUPS=barrido_final RUN_ALL_EXTRA="$(RUN_ALL_EXTRA)"

summarize:
	@$(PYTHON) scripts/summarize_results.py "$(RESULTS_DIR)"

summarize-sweep100:
	@$(PYTHON) scripts/summarize_results.py "$(SWEEP100_RESULTS_DIR)" --sort accuracy-complexity --output "$(SWEEP100_RESULTS_DIR)/summary_sweep100.md"

sweep100:
	@$(PYTHON) scripts/run_sweep100.py --make "$(MAKE)" --data-dir "$(DATA_DIR)" --results-dir "$(SWEEP100_RESULTS_DIR)" --limit "$(SWEEP100_LIMIT)" -- $(SWEEP100_EXTRA)
	@$(MAKE) summarize-sweep100

sweep100-dry:
	@$(PYTHON) scripts/run_sweep100.py --make "$(MAKE)" --data-dir "$(DATA_DIR)" --results-dir "$(SWEEP100_RESULTS_DIR)" --limit "$(SWEEP100_LIMIT)" --dry-run -- $(SWEEP100_EXTRA)

hls-config: $(HLS_CFG)

$(HLS_CFG): Makefile testbench/hls_csim_tb.cpp $(SOURCES) | $(TARGET_DIR)
	@$(PYTHON) scripts/write_hls_config.py --output "$(HLS_CFG)" --part "$(HLS_PART)" --top ff_train_kernel --cflags "$(HLS_CFLAGS)" --tb "$(THIS_DIR)/testbench/hls_csim_tb.cpp" --files "$(THIS_DIR)/src/ff_dataset.cpp" "$(THIS_DIR)/src/ff_layer.cpp" "$(THIS_DIR)/src/ff_train.cpp" "$(THIS_DIR)/src/ff_infer.cpp" "$(THIS_DIR)/include/ff_config.hpp" "$(THIS_DIR)/include/ff_types.hpp" "$(THIS_DIR)/include/ff_dataset.hpp" "$(THIS_DIR)/include/ff_layer.hpp" "$(THIS_DIR)/include/ff_train.hpp" "$(THIS_DIR)/include/ff_infer.hpp"

csim-run: $(HLS_CFG) | $(RUN_DIR)
	@$(PYTHON) scripts/run_hls_csim.py --vitis-run "$(VITIS_RUN)" --config "$(HLS_CFG)" --work-dir "$(HLS_WORK_DIR)" --log "$(HLS_CSIM_LOG)" --dataset "$(DATASET)" --train-samples "$(TRAIN_SAMPLES)" --eval-samples "$(EVAL_SAMPLES)" --epochs "$(EPOCHS)" --seed "$(SEED)" --timestamp "$(TIMESTAMP)" --variant "$(VARIANT)" --hls-part "$(HLS_PART)" --img "$(IMG_W)x$(IMG_H)" --pixel-bits "$(PIX_BITS)" --hidden-layers "$(HIDDEN_LAYERS)" --hidden-values "$(HIDDEN_VALUES)" --parallel-neurons "$(PARALLEL_NEURONS)" --threshold "$(THRESHOLD)"

# Pruebas de distintos binarios manteniendo, salvo override, la arquitectura actual.
run-bins: run-bin-16x16-1b run-bin-20x20-1b run-bin-28x28-1b run-bin-20x20-4b run-bin-20x20-6b

run-bin-16x16-1b:
	@$(MAKE) run IMG_W=16 IMG_H=16 PIX_BITS=1 DATASET=$(DATA_DIR)/mnist_16x16_1b_packed.bin

run-bin-20x20-1b:
	@$(MAKE) run IMG_W=20 IMG_H=20 PIX_BITS=1 DATASET=$(DATA_DIR)/mnist_20x20_1b_packed.bin

run-bin-28x28-1b:
	@$(MAKE) run IMG_W=28 IMG_H=28 PIX_BITS=1 DATASET=$(DATA_DIR)/mnist_28x28_1b_packed.bin

run-bin-20x20-4b:
	@$(MAKE) run IMG_W=20 IMG_H=20 PIX_BITS=4 PIXEL_SCALE=8 THRESHOLD=128 DATASET=$(DATA_DIR)/mnist_20x20_4b_packed.bin

run-bin-20x20-6b:
	@$(MAKE) run IMG_W=20 IMG_H=20 PIX_BITS=6 PIXEL_SCALE=8 THRESHOLD=128 DATASET=$(DATA_DIR)/mnist_20x20_6b_packed.bin

# Pruebas de arquitectura sobre el binario actual. Si se usa -j, estas variantes
# pueden compilarse y ejecutarse en paralelo porque escriben en carpetas distintas.
run-archs: run-arch-h32-p4 run-arch-h64-p8 run-arch-h96-p8 run-arch-h128-p8

run-arch-h32-p4:
	@$(MAKE) run HIDDEN_LAYERS=1 HIDDEN_VALUES=32 PARALLEL_NEURONS=4 THRESHOLD=56

run-arch-h64-p8:
	@$(MAKE) run HIDDEN_LAYERS=1 HIDDEN_VALUES=64 PARALLEL_NEURONS=8 THRESHOLD=72

run-arch-h96-p8:
	@$(MAKE) run HIDDEN_LAYERS=1 HIDDEN_VALUES=96 PARALLEL_NEURONS=8 THRESHOLD=80

run-arch-h128-p8:
	@$(MAKE) run HIDDEN_LAYERS=1 HIDDEN_VALUES=128 PARALLEL_NEURONS=8 THRESHOLD=96

# Barrido corto representativo. Mantiene el coste razonable y deja un .txt por corrida.
sweep:
	@$(MAKE) run IMG_W=16 IMG_H=16 PIX_BITS=1 HIDDEN_LAYERS=1 HIDDEN_VALUES=32 PARALLEL_NEURONS=4 THRESHOLD=56 DATASET=$(DATA_DIR)/mnist_16x16_1b_packed.bin
	@$(MAKE) run IMG_W=16 IMG_H=16 PIX_BITS=1 HIDDEN_LAYERS=1 HIDDEN_VALUES=64 PARALLEL_NEURONS=8 THRESHOLD=64 DATASET=$(DATA_DIR)/mnist_16x16_1b_packed.bin
	@$(MAKE) run IMG_W=20 IMG_H=20 PIX_BITS=1 HIDDEN_LAYERS=1 HIDDEN_VALUES=32 PARALLEL_NEURONS=4 THRESHOLD=64 DATASET=$(DATA_DIR)/mnist_20x20_1b_packed.bin
	@$(MAKE) run IMG_W=20 IMG_H=20 PIX_BITS=1 HIDDEN_LAYERS=1 HIDDEN_VALUES=64 PARALLEL_NEURONS=8 THRESHOLD=72 DATASET=$(DATA_DIR)/mnist_20x20_1b_packed.bin
	@$(MAKE) run IMG_W=28 IMG_H=28 PIX_BITS=1 HIDDEN_LAYERS=1 HIDDEN_VALUES=64 PARALLEL_NEURONS=8 THRESHOLD=96 DATASET=$(DATA_DIR)/mnist_28x28_1b_packed.bin
	@$(MAKE) run IMG_W=20 IMG_H=20 PIX_BITS=4 HIDDEN_LAYERS=1 HIDDEN_VALUES=64 PARALLEL_NEURONS=8 THRESHOLD=128 PIXEL_SCALE=8 DATASET=$(DATA_DIR)/mnist_20x20_4b_packed.bin

clean:
	@$(PYTHON) -c "import shutil; shutil.rmtree(r'$(BUILD_DIR)', ignore_errors=True)"

clean-runs:
	@$(PYTHON) -c "import shutil; shutil.rmtree(r'$(RUN_DIR)', ignore_errors=True)"

clean-results:
	@$(PYTHON) -c "import shutil; shutil.rmtree(r'$(RESULTS_DIR)', ignore_errors=True)"

clean-hls:
	@$(PYTHON) -c "import shutil; shutil.rmtree(r'$(HLS_WORK_ROOT)', ignore_errors=True)"
