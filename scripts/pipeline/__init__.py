"""Stage-oriented DTMB pipeline scripts for `make -j` orchestration.

Each submodule is a self-contained stage: one well-defined input set, one
well-defined output file. The Makefile at the repository root (`pipeline.mk`)
wires the stages together so incremental runs only redo the stages whose
inputs changed.
"""
