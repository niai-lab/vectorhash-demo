# Vector-HaSH GUI Demo

An interactive graphical user interface (GUI) for exploring and demonstrating **Vector-HaSH (Vector Hippocampal Scaffolded Heteroassociative Memory)**.

This project provides interactive demonstrations focused on three applications of Vector-HaSH:

* **Item Memory** — Store and retrieve individual items using the Vector-HaSH associative memory architecture.
* **Spatial Memory** — Explore how Vector-HaSH uses grid-cell-based spatial representations to encode and retrieve locations.
* **Memory Palace** — Demonstrate how spatial scaffolds can be used to associate arbitrary items with locations for structured memory and recall.

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

## Demonstrations

### Item Memory

The Item Memory demo illustrates how Vector-HaSH stores sensory items through heteroassociation with the hippocampal scaffold and subsequently recalls them from partial or corrupted inputs.

### Spatial Memory

The Spatial Memory demo visualizes how spatial locations are represented using the grid-cell scaffold and how the model can associate sensory information with locations.

### Memory Palace

The Memory Palace demo illustrates the connection between Vector-HaSH and the **method of loci**. Items can be associated with spatial landmarks and later recalled by navigating through the spatial scaffold.

This demonstrates how Vector-HaSH can use an underlying spatial representation to organize and retrieve arbitrary non-spatial information.

## Purpose

The goal of this repository is to provide an accessible, interactive way to explore selected capabilities of Vector-HaSH.

Rather than reproducing every experiment from the original paper, the GUI focuses specifically on:

**Item Memory · Spatial Memory · Memory Palace**

The interface is intended for demonstration and exploration of the model's behavior.

## Acknowledgments

This project builds upon the Vector-HaSH model and source code developed by **Sarthak Chandra, Sugandha Sharma, Rishidev Chaudhuri, and Ila Fiete**.

The underlying Vector-HaSH implementation was originally released by the **Fiete Lab**:

https://github.com/FieteLab/VectorHaSH

Please refer to and cite the original paper when using Vector-HaSH in research.

## Citation

```bibtex
@article{chandra2025episodic,
  title={Episodic and associative memory from spatial scaffolds in the hippocampus},
  author={Chandra, Sarthak and Sharma, Sugandha and Chaudhuri, Rishidev and Fiete, Ila},
  journal={Nature},
  volume={638},
  pages={739--751},
  year={2025},
  doi={10.1038/s41586-024-08392-y}
}
```
