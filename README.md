# Vector-HaSH GUI Demo

An interactive graphical user interface (GUI) for exploring and demonstrating **Vector-HaSH (Vector Hippocampal Scaffolded Heteroassociative Memory)**.

This project provides interactive demonstrations focused on three applications of Vector-HaSH:

* **Item Memory** — Illustrate how Vector-HaSH stores sensory items through heteroassociation with the hippocampal scaffold and subsequently recalls them from partial or corrupted inputs.
* **Spatial Memory** — Visualize how spatial locations are represented using the memory scaffold and how the model can associate sensory information with locations.
* **Memory Palace** — Demonstrate the connection between Vector-HaSH and the method of loci. Mnemonic items can be associated with sensory landmarks and later retrieved by noisy recalled sensory items.

## About Vector-HaSH

Vector-HaSH is a biologically inspired memory model introduced by Chandra, Sharma, Chaudhuri, and Fiete. The model uses interactions between grid-cell-like representations, hippocampal representations, and sensory information to construct a high-capacity associative memory system.

The model provides a unified framework for studying several functions associated with the hippocampal system, including associative memory, spatial memory, episodic memory, and memory palaces.

## Original Paper

This GUI is based on the Vector-HaSH model introduced in:

> Sarthak Chandra, Sugandha Sharma, Rishidev Chaudhuri, and Ila Fiete.
> **“Episodic and associative memory from spatial scaffolds in the hippocampus.”**
> *Nature*, 638, 739–751 (2025).

**Paper:**
https://doi.org/10.1038/s41586-024-08392-y

## Original Implementation

The Vector-HaSH implementation provided by the paper authors is available here:

**FieteLab/VectorHaSH:**
https://github.com/FieteLab/VectorHaSH

The Vector-HaSH code used in this project was **refactored from the original implementation provided by the paper authors** to support the interactive GUI demonstrations.

## Purpose

The goal of this repository is to provide an accessible, interactive way to explore selected capabilities of Vector-HaSH.

Rather than reproducing every experiment from the original paper, the GUI focuses specifically on:

**Item Memory · Spatial Memory · Memory Palace**

The interface is intended for demonstration and exploration of the model's behavior.

## Citation
If you use this GUI demo in your work, please cite:
```bibtex
@software{yang2026vectorhashdemo,
  author = {Yang, Jaeho and Yoon, Kijung},
  title = {Vector-HaSH GUI Demo},
  year = {2026},
  url = {https://github.com/niai-lab/vectorhash-demo},
  note = {Interactive GUI demonstration of Vector-HaSH}
}
```
