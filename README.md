# Task 2: Needleman-Wunsch Algorithm – Global Sequence Alignment

## Usage:
```bash
  python program.py --src_file sequences.txt --match 2 --mismatch -1 --gap 1 --path 1
```


## My project implements the Needleman-Wunsch algorithm for global sequence alignment of DNA sequences. 
The program:
- loads sequences from a file (FASTA or plain text) or allows manual input.
- allows setting alignment parameters (match score, mismatch penalty, and gap penalty).
- generates one optimal alignment and calculates statistics: length, matches, percentage of identical positions, and gaps.
- saves the results to a text file, creates a visualization of the alignment matrix, and generates a PDF report with results and a plot.
- allows the user to select and visualize the optimal alignment path.

## Outputs:
alignment_result.txt, alignment_plot.png, alignment_report.pdf
