## Overview
This project analyzes Whole Genome Sequencing (WGS) data from bacterial isolates to explore genetic patterns associated with **carbapenem resistance** (last-resort antibiotics). We compare resistant isolates against background genomes to identify shared vs. unique signals and generate hypotheses about resistance mechanisms.

## Research Questions (high-level)
- Which known carbapenem resistance genes (e.g., **blaKPC, blaNDM, blaVIM, blaOXA**) appear in each isolate?
- Are resistance genes likely **chromosomal** or **plasmid-borne**?
- Are there mutations in **porins**, **efflux pumps**, or **PBPs** that may contribute to reduced susceptibility?
- Do isolates share identical or distinct resistance determinants?
- Are there potentially **novel or uncommon β-lactamase variants**?

## Pipeline (current)
FASTA/WGS sequences → **Sequence embeddings (ESM-2)** → **Transformer-based modeling** → **Clustering + visualization**  
The goal is to learn representations of genomic sequence data and explore how resistant isolates group relative to background samples.

## Data
Input data are FASTA sequence files from:
- Resistant isolate dataset(s)
- Background datasets for comparison
