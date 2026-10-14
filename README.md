# Circuit Diagram Semantic Extraction

## Overview

This repository contains a proof-of-concept pipeline for extracting structured information from circuit diagrams for downstream AI applications.

The prototype focuses on application circuits and block diagrams. Functional-block illustrations are currently out of scope.

## Pipeline

The current pipeline consists of the following stages:

1. **OCR:** Extract text, component references, pin numbers, and signal labels using docTR.
2. **Component detection:** Detect supported circuit components using a fine-tuned YOLO model.
3. **IC localization:** Combine computer-vision preprocessing with VLM-based grounding to locate the main IC.
4. **Pin and connection extraction:** Use a VLM to identify pins, explicit net labels, and connected components.
5. **Output assembly:** Combine the extracted information into structured JSONL output.

## Source Code Guide

| File | Description |
|---|---|
| `main_proto.py` | Main prototype pipeline |
| `manager_ocr.py` | OCR processing |
| `manager_detector.py` | Circuit component detection |
| `manager_vlm.py` | VLM-based IC localization and pin-information extraction |
| `prompts.py` | Prompts for VLM-based extraction |
| `testing_pipeline.py` | Pipeline testing |
| `testing_ocr.py` | OCR testing |
| `test_find_IC.py` | IC localization testing |
| `testing_read_meta.py` | Reading and inspecting extracted metadata |
| `testing.py` | Component and prototype testing |

## Outputs

The main outputs for inspection are located in **`output/test_proto/`**:

- **JSONL metadata:** Structured extraction results, including OCR text, detected components, IC localization, and pin-connection information.
- **Annotated images:** Computer-vision outputs with bounding boxes, allowing visual inspection of detected regions and localization results.

These outputs illustrate the current prototype's behavior and are useful for reviewing both successful results and extraction errors.

## Development Artifacts

The `imgs/` directory contains input images used during development. Other files and output directories may contain intermediate experiments, debugging artifacts, or materials retained from prototype development.

For a focused review, start with `main_proto.py` and the relevant manager modules, then inspect `output/test_proto/` for the generated results. The remaining development artifacts are included for context and are not required for understanding the main pipeline.

## Limitations

This is an experimental proof of concept, not a production-ready system. OCR may miss or misread text, component detection can fail on unfamiliar diagram styles, and extracted pin connections may be incorrect even when the model reports high confidence. The JSONL output should be treated as a candidate representation requiring verification.

## Execution Status

The repository is provided for source-code and output inspection. It is **not currently runnable as submitted**, because the required model assets and runtime configuration have been removed.

Reproducing the pipeline requires restoring the necessary model assets, dependencies, and configuration.
