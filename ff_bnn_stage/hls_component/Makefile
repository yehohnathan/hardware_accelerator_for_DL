# Este Makefile centraliza la compilacion y la ejecucion del testbench
# host, y tambien envuelve la simulacion csim generada por Vitis HLS.

# Este Makefile se orienta al entorno cmd de Windows, que es el que usa
# make.exe en esta instalacion de Vitis.
SHELL := cmd
.SHELLFLAGS := /C

# Estas rutas describen la ubicacion del proyecto y de la instalacion de
# Vitis que se usara para compilar el testbench.
THIS_DIR := $(abspath .)
REPO_ROOT := $(abspath $(THIS_DIR)/../..)
VITIS_ROOT ?= C:/Xilinx/Vitis/2024.2
XILINX_VCXX ?= $(VITIS_ROOT)/vcxx
MINGW_BIN ?= $(VITIS_ROOT)/tps/win64/msys64/mingw64/bin
MSYS_BIN ?= $(VITIS_ROOT)/tps/win64/msys64/usr/bin
VITIS_BIN ?= $(VITIS_ROOT)/bin

# Estas variables controlan la build activa del dataset y del modelo.
IMG_W ?= 20
IMG_H ?= 20
PIX_BITS ?= 6
TRAIN_SAMPLES ?= 128
EVAL_SAMPLES ?= 32
EPOCHS ?= 2

# Este nombre sigue el convenio que ya usa el empaquetado Python.
BIN_NAME ?= mnist_$(IMG_W)x$(IMG_H)_$(PIX_BITS)b_packed.bin
BIN_PATH ?= $(REPO_ROOT)/ff_bnn_stage/mnist/data/processed/$(BIN_NAME)

# Estas rutas se usan para la build host y para el wrapper de csim.
BUILD_DIR := $(THIS_DIR)/build
HOST_EXE := $(BUILD_DIR)/train_tb_host_$(IMG_W)x$(IMG_H)_$(PIX_BITS)b.exe
CSIM_DIR := $(THIS_DIR)/hls_component/hls/csim/build
CSIM_MAKEFILE := $(CSIM_DIR)/csim.mk
CSIM_EXE := $(CSIM_DIR)/csim.exe
CSIM_BAT := $(CSIM_DIR)/sim.bat

# Estas fuentes componen el testbench host completo.
SOURCES := \
	$(THIS_DIR)/train_tb.cpp \
	$(THIS_DIR)/tb_stage_c.cpp \
	$(THIS_DIR)/tb_stage_b.cpp \
	$(THIS_DIR)/debug_utils.cpp \
	$(THIS_DIR)/forward_fw.cpp

# Este compilador host reutiliza el toolchain Mingw64 incluido en Vitis.
CXX := $(MINGW_BIN)/g++.exe

# Estas banderas replican el entorno minimo necesario para usar ap_int,
# ap_fixed y las cabeceras de simulacion de Vitis.
HOST_DEFINES := \
	-DFF_IMAGE_WIDTH_CFG=$(IMG_W) \
	-DFF_IMAGE_HEIGHT_CFG=$(IMG_H) \
	-DFF_INPUT_BITS_PER_PIXEL_CFG=$(PIX_BITS) \
	-D__HLS_CSIM__ \
	-D__VITIS_HLS__ \
	-D__SIM_FPO__ \
	-D__SIM_FFT__ \
	-D__SIM_FIR__ \
	-D__SIM_DDS__ \
	-D__DSP48E1__ \
	-DNT

HOST_INCLUDES := \
	-I"$(VITIS_ROOT)/include" \
	-I"$(VITIS_ROOT)/include/ap_sysc" \
	-I"$(VITIS_ROOT)/common/technology/generic/SystemC" \
	-I"$(VITIS_ROOT)/common/technology/generic/SystemC/AESL_FP_comp" \
	-I"$(VITIS_ROOT)/common/technology/generic/SystemC/AESL_comp" \
	-I"$(VITIS_ROOT)/win64/tools/auto_cc/include"

HOST_CXXFLAGS := \
	-std=gnu++14 \
	-g \
	-Wno-unknown-pragmas \
	-Wno-abi \
	$(HOST_DEFINES) \
	$(HOST_INCLUDES)

.PHONY: all help print-config host run csim csim-run clean clean-csim

# La build host se deja como objetivo por defecto para tener una via rapida
# de compilacion fuera del flujo completo de HLS.
all: host

# Esta ayuda resume las variables y los targets mas utiles para el trabajo
# diario con distintas resoluciones y distintos empaquetados.
help:
	@echo Targets disponibles:
	@echo   make host
	@echo   make run
	@echo   make csim
	@echo   make csim-run
	@echo   make clean
	@echo.
	@echo Variables configurables:
	@echo   IMG_W=20 IMG_H=20 PIX_BITS=6
	@echo   TRAIN_SAMPLES=128 EVAL_SAMPLES=32 EPOCHS=2
	@echo   BIN_PATH=$(REPO_ROOT)/ff_bnn_stage/mnist/data/processed/...
	@echo.
	@echo Ejemplos:
	@echo   make run IMG_W=16 IMG_H=16 PIX_BITS=4
	@echo   make csim-run IMG_W=28 IMG_H=28 PIX_BITS=1 EPOCHS=1

# Este target imprime la configuracion efectiva antes de compilar o ejecutar.
print-config:
	@echo THIS_DIR       = $(THIS_DIR)
	@echo REPO_ROOT      = $(REPO_ROOT)
	@echo VITIS_ROOT     = $(VITIS_ROOT)
	@echo IMG_W          = $(IMG_W)
	@echo IMG_H          = $(IMG_H)
	@echo PIX_BITS       = $(PIX_BITS)
	@echo BIN_NAME       = $(BIN_NAME)
	@echo BIN_PATH       = $(BIN_PATH)
	@echo TRAIN_SAMPLES  = $(TRAIN_SAMPLES)
	@echo EVAL_SAMPLES   = $(EVAL_SAMPLES)
	@echo EPOCHS         = $(EPOCHS)
	@echo HOST_EXE       = $(HOST_EXE)
	@echo CSIM_EXE       = $(CSIM_EXE)

# Este directorio agrupa los binarios host creados manualmente.
$(BUILD_DIR):
	@if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"

# Esta receta compila el testbench host con la geometria del dataset elegida.
$(HOST_EXE): $(SOURCES) | $(BUILD_DIR)
	@set "XILINX_VCXX=$(XILINX_VCXX)" && \
	set "PATH=$(MSYS_BIN);$(MINGW_BIN);$(VITIS_BIN);%PATH%" && \
	"$(CXX)" $(HOST_CXXFLAGS) $(SOURCES) -o "$(HOST_EXE)"

# Este alias compila el ejecutable host configurado con las variables make.
host: $(HOST_EXE)

# Este target ejecuta el binario host y le pasa el dataset y los limites
# del experimento mediante variables de entorno.
run: $(HOST_EXE)
	@set "XILINX_VCXX=$(XILINX_VCXX)" && \
	set "PATH=$(MSYS_BIN);$(MINGW_BIN);$(VITIS_BIN);%PATH%" && \
	set "FF_INPUT_BIN_PATH=$(BIN_PATH)" && \
	set "FF_TRAIN_SAMPLES=$(TRAIN_SAMPLES)" && \
	set "FF_EVAL_SAMPLES=$(EVAL_SAMPLES)" && \
	set "FF_EPOCHS=$(EPOCHS)" && \
	"$(HOST_EXE)"

# Este target recompila csim con la misma configuracion elegida para host.
csim:
	@if not exist "$(CSIM_MAKEFILE)" ( \
		echo ERROR: no existe $(CSIM_MAKEFILE). && \
		echo ERROR: primero hay que generar la solucion HLS de Vitis. && \
		exit /b 1 \
	)
	@set "XILINX_VCXX=$(XILINX_VCXX)" && \
	set "PATH=$(MSYS_BIN);$(MINGW_BIN);$(VITIS_BIN);%PATH%" && \
	"$(MINGW_BIN)/make.exe" -C "$(CSIM_DIR)" -f csim.mk clean all \
	CCFLAG="-DFF_IMAGE_WIDTH_CFG=$(IMG_W) \
	-DFF_IMAGE_HEIGHT_CFG=$(IMG_H) \
	-DFF_INPUT_BITS_PER_PIXEL_CFG=$(PIX_BITS)"

# Este target ejecuta csim con el dataset y los limites del experimento.
csim-run: csim
	@if not exist "$(CSIM_BAT)" ( \
		echo ERROR: no existe $(CSIM_BAT). && \
		exit /b 1 \
	)
	@set "XILINX_VCXX=$(XILINX_VCXX)" && \
	set "PATH=$(MSYS_BIN);$(MINGW_BIN);$(VITIS_BIN);%PATH%" && \
	set "FF_INPUT_BIN_PATH=$(BIN_PATH)" && \
	set "FF_TRAIN_SAMPLES=$(TRAIN_SAMPLES)" && \
	set "FF_EVAL_SAMPLES=$(EVAL_SAMPLES)" && \
	set "FF_EPOCHS=$(EPOCHS)" && \
	cd /d "$(CSIM_DIR)" && \
	sim.bat

# Este target limpia solo los binarios host creados por este Makefile.
clean:
	@if exist "$(BUILD_DIR)" rmdir /S /Q "$(BUILD_DIR)"

# Este target limpia la build generada por csim sin tocar la solucion HLS.
clean-csim:
	@if not exist "$(CSIM_MAKEFILE)" exit /b 0
	@set "XILINX_VCXX=$(XILINX_VCXX)" && \
	set "PATH=$(MSYS_BIN);$(MINGW_BIN);$(VITIS_BIN);%PATH%" && \
	"$(MINGW_BIN)/make.exe" -C "$(CSIM_DIR)" -f csim.mk clean
