"""
Generate Synthetic FASTA Test Data
==================================
Creates realistic synthetic bacterial sequences for testing the pipeline.

This generates three isolate files with:
- Varying GC content (typical for different bacterial species)
- Embedded resistance gene motifs at different frequencies
- Realistic sequence lengths and read counts

Usage:
    python generate_test_data.py
"""

import os
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "data"
READS_PER_FILE = 5000          # Number of sequences per isolate
READ_LENGTH = 250              # Typical Illumina read length
RANDOM_SEED = 42

# Isolate profiles with different characteristics
ISOLATE_PROFILES = {
    "dataset.fasta": {
        "gc_content": 0.52,           # ~52% GC (typical Klebsiella)
        "resistance_genes": {
            "blaKPC": 0.15,           # 15% of reads contain KPC motifs
            "blaNDM": 0.08,           # 8% contain NDM motifs
            "blaOXA": 0.05,           # 5% contain OXA motifs
        },
        "description": "Main isolate - high KPC, moderate NDM"
    },
    "K22_sequence.fasta": {
        "gc_content": 0.50,           # ~50% GC
        "resistance_genes": {
            "blaKPC": 0.03,           # Low KPC
            "blaNDM": 0.20,           # High NDM
            "blaVIM": 0.10,           # Some VIM
        },
        "description": "Background 1 - high NDM, some VIM"
    },
    "K31_sequence.fasta": {
        "gc_content": 0.55,           # ~55% GC
        "resistance_genes": {
            "blaOXA": 0.25,           # High OXA
            "blaVIM": 0.05,           # Low VIM
        },
        "description": "Background 2 - high OXA"
    }
}

# Resistance gene motif sequences (expanded from main pipeline)
RESISTANCE_MOTIFS = {
    "blaKPC": [
        "TGGCGCCGGTTGC",   # KPC active site region
        "CGTGGTACGGCA",    # KPC conserved motif
        "ATGGCACTGTCA",    # Upstream region
    ],
    "blaNDM": [
        "GGGCGCTGCGAT",    # NDM zinc-binding motif
        "GATCGCAATGGT",    # NDM conserved region
        "CACCGCATGTCT",    # Metallo-β-lactamase signature
    ],
    "blaOXA": [
        "ACGAATGCCTGA",    # OXA-48 signature
        "TCGACTGGCAAT",    # OXA conserved motif
        "GTCAAGCTGGGC",    # Carbapenemase domain
    ],
    "blaVIM": [
        "GCGCGGTGAACT",    # VIM active site
        "CGCGGAATACGC",    # VIM conserved region
        "TGCGGTGATCCA",    # Metallo-β-lactamase motif
    ]
}

# =============================================================================
# SEQUENCE GENERATION FUNCTIONS
# =============================================================================

def generate_random_sequence(length, gc_content):
    """
    Generate a random DNA sequence with specified GC content.
    
    Args:
        length: Desired sequence length
        gc_content: Target GC proportion (0.0 to 1.0)
    
    Returns:
        Random DNA string
    """
    # Calculate base probabilities
    gc_prob = gc_content / 2      # Split between G and C
    at_prob = (1 - gc_content) / 2  # Split between A and T
    
    bases = ['A', 'T', 'G', 'C']
    weights = [at_prob, at_prob, gc_prob, gc_prob]
    
    return ''.join(random.choices(bases, weights=weights, k=length))

def embed_motif(sequence, motif, position=None):
    """
    Embed a resistance motif into a sequence at a given position.
    
    Args:
        sequence: Original DNA sequence
        motif: Motif sequence to embed
        position: Where to embed (None = random position)
    
    Returns:
        Modified sequence with embedded motif
    """
    seq_list = list(sequence)
    
    if position is None:
        # Random position that fits the motif
        max_pos = len(sequence) - len(motif)
        if max_pos <= 0:
            return sequence
        position = random.randint(0, max_pos)
    
    # Embed the motif
    for i, base in enumerate(motif):
        if position + i < len(seq_list):
            seq_list[position + i] = base
    
    return ''.join(seq_list)

def generate_read(gc_content, resistance_genes):
    """
    Generate a single sequencing read.
    
    Args:
        gc_content: Target GC content for background sequence
        resistance_genes: Dict of {gene_name: probability}
    
    Returns:
        Tuple of (sequence, embedded_genes)
    """
    # Start with random background sequence
    sequence = generate_random_sequence(READ_LENGTH, gc_content)
    embedded_genes = []
    
    # Possibly embed resistance motifs
    for gene, probability in resistance_genes.items():
        if random.random() < probability:
            # Choose a random motif for this gene
            motif = random.choice(RESISTANCE_MOTIFS[gene])
            sequence = embed_motif(sequence, motif)
            embedded_genes.append(gene)
    
    return sequence, embedded_genes

def generate_fasta_file(filepath, profile, num_reads):
    """
    Generate a complete FASTA file for one isolate.
    
    Args:
        filepath: Output file path
        profile: Isolate profile dict with gc_content and resistance_genes
        num_reads: Number of reads to generate
    """
    gene_counts = {gene: 0 for gene in RESISTANCE_MOTIFS.keys()}
    
    with open(filepath, 'w') as f:
        for i in range(num_reads):
            # Generate read
            sequence, embedded = generate_read(
                profile["gc_content"],
                profile.get("resistance_genes", {})
            )
            
            # Track embedded genes
            for gene in embedded:
                gene_counts[gene] += 1
            
            # Write FASTA entry
            header = f">read_{i+1}"
            if embedded:
                header += f" genes={','.join(embedded)}"
            
            f.write(f"{header}\n")
            
            # Write sequence in lines of 80 characters
            for j in range(0, len(sequence), 80):
                f.write(f"{sequence[j:j+80]}\n")
    
    return gene_counts

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all test FASTA files."""
    random.seed(RANDOM_SEED)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING SYNTHETIC TEST DATA")
    print("=" * 60)
    print()
    
    for filename, profile in ISOLATE_PROFILES.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        print(f"Generating: {filename}")
        print(f"  Description: {profile['description']}")
        print(f"  Target GC: {profile['gc_content']:.0%}")
        print(f"  Resistance genes: {list(profile.get('resistance_genes', {}).keys())}")
        
        gene_counts = generate_fasta_file(filepath, profile, READS_PER_FILE)
        
        print(f"  Reads generated: {READS_PER_FILE:,}")
        print(f"  Embedded gene counts: {gene_counts}")
        print()
    
    print("=" * 60)
    print("TEST DATA GENERATION COMPLETE")
    print("=" * 60)
    print()
    print("Files created:")
    for filename in ISOLATE_PROFILES.keys():
        filepath = os.path.join(OUTPUT_DIR, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {filepath} ({size_kb:.1f} KB)")
    
    print()
    print("Next step: Run the analysis pipeline:")
    print("  python carbapenem_resistance_analysis.py")

if __name__ == "__main__":
    main()