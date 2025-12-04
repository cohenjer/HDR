import os
import re
from pathlib import Path
from collections import defaultdict

def find_markdown_files(directories):
    """Find all markdown files in the given directories."""
    md_files = []
    for directory in directories:
        path = Path(directory)
        if path.exists():
            md_files.extend(path.rglob("*.md"))
    return sorted(md_files)

def extract_acronyms(text):
    """Extract potential acronyms (2-5 uppercase letters)."""
    # Match acronyms: 2-5 consecutive uppercase letters, possibly with hyphens
    pattern = r'\b[A-Z]{2,5}(?:-[A-Z]{2,5})*\b'
    return set(re.findall(pattern, text))

def extract_hyphenated_words(text):
    """Extract words that might have ambiguous hyphenation."""
    # Match hyphenated compound words
    pattern = r'\b\w+(?:-\w+)+\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Filter out common patterns we want to keep
    filtered = []
    for match in matches:
        # Skip markdown links and common patterns
        if not match.startswith('http') and not match.startswith('--'):
            filtered.append(match.lower())
    
    return set(filtered)

def extract_technical_terms(text):
    """Extract technical terms from a predefined list."""
    # Comprehensive list of technical terms to search for
    technical_terms_list = [
        # Decomposition methods
        "tensor decomposition", "Tucker decomposition", "PARAFAC2",
        "matrix factorization", "canonical polyadic decomposition",
        
        # Optimization and algorithms
        "sparse coding", "dictionary learning", "identifiability",
        "convergence rates", "extrapolation", "inertial algorithms",
        "proximal operators", "majorant", "optimization", "regularization",
        "constraints", "alternating optimization", "block coordinate descent",
        "alternating least squares", "multiplicative update",
        "hierarchical alternating least squares",
        
        # Representations
        "sparse representation", "low separation rank", "Kronecker product",
        "Khatri-Rao product", "Frobenius norm", "beta-divergence",
        "generalized low-rank models", "low-rank approximation",
        "nonnegative matrix factorization",
        
        # Applications
        "hyperspectral unmixing", "spectral unmixing", "fluorescence spectroscopy",
        "remote sensing", "chemometrics", "bilinear models",
        "multiway data analysis", "dimensionality reduction",
        "blind source separation", "topic modeling", "image denoising",
        "automatic music transcription",
        
        # Data types
        "temporal data", "multimodal data", "biomarkers", "sparse tensor",
        "dense matrix", "factor matrices", "coupled factorization",
        "hyperspectral data", "spectral data",
        
        # Dictionary-based
        "separable dictionary", "overcomplete dictionary", "multiple dictionaries",
        "dictionary-based low-rank approximations",
        
        # Imaging
        "single-pixel imaging", "compressive sensing", "spatial multiplexing",
        "computational imaging", "inverse problem", "Poisson-Gaussian mixture",
        "Hadamard patterns", "Hadamard matrices", "spectral cameras",
        "hyperspectral imaging", "image reconstruction",
        
        # Noise models
        "Poisson noise", "Gaussian noise", "Poisson distribution",
        
        # Other technical concepts
        "photon count", "signal to noise ratio", "signal strength",
        "binary mask", "vectorize", "spatial dimension",
    ]
    
    found_terms = set()
    text_lower = text.lower()
    
    # Search for each technical term (case-insensitive)
    for term in technical_terms_list:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            found_terms.add(term)
    
    return found_terms

def get_relative_path(file_path, base_dir="./"):
    """Get relative path from base directory."""
    try:
        rel_path = os.path.relpath(file_path, base_dir)
        return rel_path
    except ValueError:
        return str(file_path)

def create_markdown_link(file_path, base_dir="./"):
    """Create a Jupyter Book compatible markdown link."""
    rel_path = get_relative_path(file_path, base_dir)
    # Remove .md extension for Jupyter Book
    link_path = rel_path.replace('.md', '')
    # The file will be located in ./intro/* and therefore the path to other files should be adjusted by going up one level
    link_path = os.path.join('..', link_path)
    # Create a readable label from the file name
    label = Path(file_path).stem
    # uppercase the label
    label = label.upper()
    return f"[[{label}]]({link_path})"

def analyze_files(directories, base_dir="./"):
    """Analyze all markdown files and collect terms with locations."""
    md_files = find_markdown_files(directories)
    
    acronyms = defaultdict(list)
    technical_terms = defaultdict(list)
    hyphenated_words = defaultdict(list)
    
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            link = create_markdown_link(file_path, base_dir)
            
            # Extract acronyms
            file_acronyms = extract_acronyms(content)
            for acronym in file_acronyms:
                acronyms[acronym].append(link)
            
            # Extract hyphenated words
            file_hyphenated = extract_hyphenated_words(content)
            for word in file_hyphenated:
                hyphenated_words[word].append(link)
            
            # Extract technical terms
            file_terms = extract_technical_terms(content)
            for term in file_terms:
                technical_terms[term.lower()].append(link)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return acronyms, technical_terms, hyphenated_words

def format_output(acronyms, technical_terms, hyphenated_words, output_file="terminology_list.md"):
    """Format the output as markdown."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Terminology and Acronyms Reference\n\n")
        f.write("This document lists all acronyms, technical terms, and words with ambiguous spelling found in the manuscript.\n\n")
        
        # Acronyms
        f.write("## Acronyms\n\n")
        for acronym in sorted(acronyms.keys()):
            locations = list(set(acronyms[acronym]))  # Remove duplicates
            f.write(f"- **{acronym}**")
            for loc in sorted(locations):
                f.write(f" {loc}")
            f.write("\n")
        
        # Technical Terms
        f.write("## Technical Terms\n\n")
        for term in sorted(technical_terms.keys()):
            locations = list(set(technical_terms[term]))
            f.write(f"- **{term}**\n")
            for loc in sorted(locations): 
                f.write(f" {loc}")
            f.write("\n")
        
        # Hyphenated words
        f.write("## Words with Ambiguous Spelling (Hyphenated)\n\n")
        for word in sorted(hyphenated_words.keys()):
            locations = list(set(hyphenated_words[word]))
            f.write(f"- **{word}**\n")
            for loc in sorted(locations):  # Limit to first 3 locations
                f.write(f" {loc}")
            f.write("\n")

def main():
    """Main function."""
    # Define directories to scan
    directories = ["./part1", "./part2", "./part3"]
    base_dir = "./"
    
    print("Scanning markdown files...")
    acronyms, technical_terms, hyphenated_words = analyze_files(directories, base_dir)
    
    print(f"Found {len(acronyms)} unique acronyms")
    print(f"Found {len(technical_terms)} unique technical terms")
    print(f"Found {len(hyphenated_words)} unique hyphenated words")
    
    print("\nGenerating markdown output...")
    format_output(acronyms, technical_terms, hyphenated_words)
    
    print("Done! Output saved to terminology_list.md")

if __name__ == "__main__":
    main()