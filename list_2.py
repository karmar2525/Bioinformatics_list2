# %%
"""
Task 2: Needleman-Wunsch Algorithm – Global Sequence Alignment

Usage:
  python program.py --src_file sequences.txt --match 2 --mismatch -1 --gap 1 --path 1

Parameters:
  --src_file   file with two sequences (FASTA or plain text)
  --match      score for a match (default: 1)
  --mismatch   penalty for a mismatch (default: -1)
  --gap        penalty for a gap (default: 1)
  --path       index of the optimal path to use (default: 1)

Output:
  alignment_result.txt, alignment_plot.png, alignment_report.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from fpdf import FPDF
from Bio import SeqIO


# %%
def validate_sequence(seq):
    """
    Validates if the given sequence contains only valid characters: A, C, G, T.

    Args:
        seq (str): The sequence to validate.

    Returns:
        bool: True if the sequence contains only valid characters, otherwise False.
    """
    valid_chars = {'A', 'C', 'G', 'T'}
    return all(char in valid_chars for char in seq)
#%%
# Function to read two sequences from a file (FASTA or plain text)
def read_sequence_from_file(file_path):
    """
       Reads two DNA sequences from a FASTA file or prompts the user for manual input if no file is provided.

       The function attempts to parse up to two sequences from a given FASTA file. If the file is missing
       or not provided, it requests the sequences manually via console input. The sequences are validated to ensure
       they contain only valid nucleotide characters: A, T, C, G (case-insensitive).

       Args:
           file_path (str): Path to the FASTA file containing at least two DNA sequences. If None, manual input is used.

       Returns:
           tuple: A tuple containing two uppercase DNA sequences (seq_a, seq_b).

       Raises:
           ValueError: If fewer than two sequences are found in the file or the sequences contain invalid characters.
           Exception: If an error occurs while reading the file.
       """

    if file_path:
        sequences = []
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append(str(record.seq))
                if len(sequences) >= 2:
                    break

            if len(sequences) < 2:
                raise ValueError("FASTA file must contain at least two sequences.")
            a, b = sequences[:2]
        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            raise
    else:
        a, b = manual_input()

    if not validate_sequence(a) or not validate_sequence(b):
        raise ValueError("Sequences must only contain the letters A, T, C, G.")
    return a.upper(), b.upper()

# %%
# Manual input fallback
def manual_input():
    """
    Prompts the user to manually enter two sequences when the file input is unavailable.
    The function requests the sequences one after the other and returns them as a tuple.

    Returns:
        tuple: A tuple containing the manually inputted sequences as strings (seq_a, seq_b).
    """
    print("Sequences were not provided in a file. Please enter them manually.")
    while True:
        seq_a = input("Sequence A: ").strip()
        seq_b = input("Sequence B: ").strip()

        # Validate sequences
        if not validate_sequence(seq_a) or not validate_sequence(seq_b):
            print("Invalid sequence. Sequences must only contain the characters A, C, G, T.")
            continue

        return seq_a, seq_b


# %%
# Initialize matrix with gap penalties
def initialize(matrix, gap_penalty):
    """
    Initializes the first row and the first column of the scoring matrix with cumulative gap penalties.
    This prepares the matrix for dynamic programming by setting the edge values based on the gap penalty.

    Args:
        matrix (numpy.ndarray): The score matrix to be initialized.
        gap_penalty (int): The penalty for introducing a gap in the sequence alignment.

    Returns:
        numpy.ndarray: The initialized score matrix.
    """
    for i in range(1, matrix.shape[0]):
        matrix[i, 0] = matrix[i - 1, 0] - gap_penalty
    for j in range(1, matrix.shape[1]):
        matrix[0, j] = matrix[0, j - 1] - gap_penalty
    return matrix

# %%
# Scoring function
def score(a, b, match=1, mismatch=-1):
    """
    Returns the score for aligning two characters based on the match/mismatch scoring system.

    Args:
        a (str): The first character in the alignment.
        b (str): The second character in the alignment.
        match (int, optional): The score for a match (default is 1).
        mismatch (int, optional): The penalty for a mismatch (default is -1).

    Returns:
        int: The score for aligning characters a and b.
    """
    return match if a == b else mismatch

# %%
# Fill score matrix
def fill_matrix(seq_a, seq_b, gap_penalty, match, mismatch):
    """
    Fills the scoring matrix using dynamic programming. The matrix is populated based on the sequence
    alignment rules (match, mismatch, and gap penalties).

    Args:
        seq_a (str): The first sequence to align.
        seq_b (str): The second sequence to align.
        gap_penalty (int): The penalty for introducing a gap in the alignment.
        match (int): The score for matching characters.
        mismatch (int): The penalty for mismatched characters.

    Returns:
        numpy.ndarray: The completed scoring matrix.
    """
    matrix = np.zeros((len(seq_a)+1, len(seq_b)+1))
    matrix = initialize(matrix, gap_penalty)
    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            matrix[i, j] = max(
                matrix[i-1, j-1] + score(seq_a[i-1], seq_b[j-1], match, mismatch),
                matrix[i-1, j] - gap_penalty,
                matrix[i, j-1] - gap_penalty
            )
    return matrix

# %%
# Traceback path
def traceback(seq_a, seq_b, matrix, gap_penalty, match, mismatch, pick_path=1):
    """
    Traces back through the scoring matrix to extract the optimal alignment path.
    Multiple paths may be available; the function returns the chosen one.

    Args:
        seq_a (str): The first sequence to align.
        seq_b (str): The second sequence to align.
        matrix (numpy.ndarray): The scoring matrix from dynamic programming.
        gap_penalty (int): The penalty for a gap.
        match (int): The score for matching characters.
        mismatch (int): The penalty for mismatching characters.
        pick_path (int, optional): The index of the optimal alignment path to select (default is 1).

    Returns:
        tuple: A tuple containing:
            - Aligned sequence A (str)
            - Aligned sequence B (str)
            - List of coordinates for the selected alignment path
            - List of all optimal alignment paths
    """
    paths = all_alignment_paths(matrix, seq_a, seq_b, gap_penalty, match, mismatch)
    chosen = paths[pick_path-1] if pick_path <= len(paths) else paths[0]
    a = ''.join(x for x, _ in chosen)
    b = ''.join(y for _, y in chosen)
    coords = compute_path_coords(chosen, seq_a, seq_b)
    return a, b, coords, paths

# %%
# Path coords for plotting
def compute_path_coords(alignment_path, seq_a, seq_b):
    """
    Converts an alignment path (list of tuples) to a series of coordinates
    in the scoring matrix for plotting purposes.

    Args:
        alignment_path (list): A list of tuples representing the alignment path.
        seq_a (str): The first sequence to align.
        seq_b (str): The second sequence to align.

    Returns:
        list: A list of coordinates corresponding to the alignment path.
    """
    coords = []
    i, j = len(seq_a), len(seq_b)
    for a, b in reversed(alignment_path):
        coords.append((i, j))
        if a != '-' and b != '-':
            i -= 1
            j -= 1
        elif a == '-' and b != '-':
            j -= 1
        elif b == '-' and a != '-':
            i -= 1
    coords.append((0, 0))
    return coords[::-1]

# %%
# All optimal paths
def all_alignment_paths(matrix, seq_a, seq_b, gap_penalty, match, mismatch):
    """
    Finds all optimal alignment paths by performing a depth-first search on the scoring matrix.
    This allows the retrieval of multiple equally optimal alignments.

    Args:
        matrix (numpy.ndarray): The scoring matrix.
        seq_a (str): The first sequence to align.
        seq_b (str): The second sequence to align.
        gap_penalty (int): The penalty for introducing a gap.
        match (int): The score for matching characters.
        mismatch (int): The penalty for mismatching characters.

    Returns:
        list: A list of all optimal alignment paths, each as a list of tuples.
    """
    stack = [(len(seq_a), len(seq_b), [])]
    paths = []
    while stack:
        i, j, path = stack.pop()
        if i == 0 and j == 0:
            paths.append(path[::-1])
            continue
        if i > 0 and matrix[i, j] == matrix[i - 1, j] - gap_penalty:
            stack.append((i - 1, j, path + [(seq_a[i - 1], '-')]))
        if j > 0 and matrix[i, j] == matrix[i, j - 1] - gap_penalty:
            stack.append((i, j - 1, path + [('-', seq_b[j - 1])]))
        if i > 0 and j > 0 and matrix[i, j] == matrix[i - 1, j - 1] + score(seq_a[i - 1], seq_b[j - 1], match, mismatch):
            stack.append((i - 1, j - 1, path + [(seq_a[i - 1], seq_b[j - 1])]))
    return paths

# %%
# Plot score matrix
def plot_matrix(matrix, seq_a, seq_b, path_coords, title='Score Matrix', alignment_path_text=''):
    """
    Plots the scoring matrix with the alignment path overlayed. The matrix is visualized using a heatmap.
    The optimal alignment path is shown in pink.

    Args:
        matrix (numpy.ndarray): The scoring matrix.
        seq_a (str): The first sequence.
        seq_b (str): The second sequence.
        path_coords (list): A list of coordinates that represent the alignment path.
        title (str, optional): The title of the plot (default is 'Score Matrix').
        alignment_path_text (str, optional): Additional text to display on the plot (default is empty).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score", rotation=270, labelpad=15)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{int(matrix[i, j])}', ha='center', va='center', color='black', fontsize=8)
    ax.set_xticks(np.arange(len(seq_b)+1))
    ax.set_yticks(np.arange(len(seq_a)+1))
    ax.set_xticklabels([' '] + list(seq_b))
    ax.set_yticklabels([' '] + list(seq_a))
    ax.set_xlabel("Sequence B")
    ax.set_ylabel("Sequence A")
    path_y, path_x = zip(*path_coords)
    ax.plot(path_x, path_y, color='deeppink', linewidth=2, marker='o', markersize=4, label="Optimal Path")
    ax.legend(loc='upper left')
    plt.title(title)
    ax.text(0.5, -0.15, alignment_path_text, ha='center', va='top', transform=ax.transAxes, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("alignment_plot.png", dpi=300)
    plt.show()

# %%
# Choose and plot path
def select_and_plot_path(matrix, seq_a, seq_b, all_paths):
    """
    Allows the user to select an alignment path from the available options
    and visualizes the chosen path on the scoring matrix plot.

    Args:
        matrix (numpy.ndarray): The scoring matrix.
        seq_a (str): The first sequence.
        seq_b (str): The second sequence.
        all_paths (list): List of all optimal alignment paths.
    """
    print(f"Available paths: {len(all_paths)}")
    while True:
        try:
            path_index = int(input("Enter the path number to visualize (1 based index): "))
            if 1 <= path_index <= len(all_paths):
                break
            else:
                print("Invalid path number. Please enter a number between 1 and", len(all_paths))
        except ValueError:
            print("Invalid input. Please enter a number.")
    selected_path_coords = compute_path_coords(all_paths[path_index - 1], seq_a, seq_b)
    plot_matrix(matrix, seq_a, seq_b, selected_path_coords, f"Selected Alignment Path (Path {path_index})", "")

#%%
# PDF report
def generate_pdf(seq_a, seq_b, alignment_a, alignment_b, stats, path_label, all_paths, first_path_coords):
    """
    Generates a PDF report containing the alignment results, including the sequences,
    aligned sequences, alignment statistics, other optimal paths, and the plot.

    Args:
        seq_a (str): The first sequence.
        seq_b (str): The second sequence.
        alignment_a (str): The aligned version of sequence A.
        alignment_b (str): The aligned version of sequence B.
        stats (str): Alignment statistics (e.g., length, matches, gaps).
        path_label (str): Label for the selected alignment path.
        all_paths (list): List of all optimal alignment paths.
        first_path_coords (list): Coordinates of the first optimal path.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(200, 10, txt="Needleman-Wunsch Global Alignment Report", ln=True, align='C')
    pdf.ln(4)
    pdf.set_font("Arial", style='B', size=9)
    pdf.cell(0, 10, txt=f"Alignment Path: {path_label}", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 8, txt=f"Sequence A: {seq_a}")
    pdf.multi_cell(0, 8, txt=f"Sequence B: {seq_b}")
    pdf.ln(2)
    pdf.set_font("Arial", style='B', size=9)
    pdf.cell(0, 8, txt="Aligned Sequences:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, txt=f"Aligned A: {alignment_a}")
    pdf.multi_cell(0, 8, txt=f"Aligned B: {alignment_b}")
    pdf.ln(2)
    pdf.set_font("Arial", style='B', size=9)
    pdf.cell(0, 8, txt="Alignment Statistics:", ln=True)
    pdf.set_font("Arial", size=9)
    pdf.multi_cell(0, 8, txt=f"{stats}")
    pdf.ln(2)
    pdf.set_font("Arial", style='B', size=9)
    pdf.cell(0, 8, txt="Other Optimal Alignment Paths:", ln=True)
    pdf.set_font("Arial", size=8)
    for idx, path in enumerate(all_paths, 1):
        pdf.multi_cell(0, 9, txt=f"Path {idx}: {''.join(x for x, _ in path)} / {''.join(y for _, y in path)}")
    pdf.ln(2)
    pdf.image("alignment_plot.png", x=10, w=150)
    pdf.output("alignment_report.pdf")


# %%
# Save alignment results to a text file
def save_alignment_to_file(alignment_a, alignment_b, match, gaps, identity):
    """
    Saves the alignment result to a text file.
    Args:
        alignment_a (str): Aligned sequence A.
        alignment_b (str): Aligned sequence B.
        match (int): The number of matching positions.
        gaps (int): The number of gaps in the alignment.
        identity (float): Percentage of matching positions.
    """
    with open("alignment_result.txt", "w") as file:
        file.write("Needleman-Wunsch Global Alignment Result\n")
        file.write(f"Sequence A: {alignment_a}\n")
        file.write(f"Sequence B: {alignment_b}\n")
        file.write("\n")
        file.write(f"Length: {len(alignment_a)}, Matches: {match} ({identity:.2f}%), Gaps: {gaps}\n")

#%%
# Main
def main():
    """
    Main function that handles argument parsing, sequence alignment,
    result saving, visualization, and PDF report generation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', help="File with sequences (two lines or FASTA)")
    parser.add_argument('--match', type=int, default=1)
    parser.add_argument('--mismatch', type=int, default=-1)
    parser.add_argument('--gap', type=int, default=1)
    parser.add_argument('--path', type=int, default=1)
    args = parser.parse_args()

    seq_a, seq_b = read_sequence_from_file(args.src_file) if args.src_file and os.path.exists(args.src_file) else manual_input()
    matrix = fill_matrix(seq_a, seq_b, args.gap, args.match, args.mismatch)
    alignment_a, alignment_b, path_coords, all_paths = traceback(seq_a, seq_b, matrix, args.gap, args.match, args.mismatch, args.path)

    match = sum(1 for x, y in zip(alignment_a, alignment_b) if x == y)
    gaps = alignment_a.count('-') + alignment_b.count('-')
    identity = match / len(alignment_a) * 100
    stats = f"Length: {len(alignment_a)}, Matches: {match} ({identity:.2f}%), Gaps: {gaps}"

    save_alignment_to_file(alignment_a, alignment_b, match, gaps, identity)
    generate_pdf(seq_a, seq_b, alignment_a, alignment_b, stats, f"Path {args.path}", all_paths, path_coords)
    select_and_plot_path(matrix, seq_a, seq_b, all_paths)

if __name__ == "__main__":
    main()
