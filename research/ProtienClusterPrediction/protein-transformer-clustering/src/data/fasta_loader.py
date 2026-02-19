import os

def load_fasta_sequences(fasta_file):
    sequences = []
    with open(fasta_file, 'r') as file:
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences

def load_all_fasta_sequences(directory):
    all_sequences = {}
    for filename in os.listdir(directory):
        if filename.endswith(".fasta"):
            file_path = os.path.join(directory, filename)
            all_sequences[filename] = load_fasta_sequences(file_path)
    return all_sequences