# English-Lyrics-Generator

An English lyrics generator trained on a subset of the Genius website using seq2seq technique.

---

## Project Overview

This project is a Natural Language Processing (NLP) application that builds a dataset of English pop song lyrics, preprocesses and tokenizes the text, constructs a vocabulary, and prepares data loaders and a model for sequence-to-sequence lyrics generation. The core logic is implemented in the `englishlyricsgen.ipynb` notebook.

---

## Notebook Structure & Workflow

Below is a detailed breakdown of the notebook's stages:

### 1. Getting the Data

- **Data Download:**  
  The dataset `genius-song-lyrics-with-language-information` is downloaded using `kagglehub` and moved for local usage.
  ```python
  import kagglehub
  import shutil
  path = kagglehub.dataset_download("carlosgdcj/genius-song-lyrics-with-language-information")
  shutil.move(path, r'/kaggle/working/data')
  ```
- **Dependencies Installation:**  
  Installs required libraries such as `fireducks` (for efficient pandas operations) and `gdown` (for downloading from Google Drive if needed).

### 2. Imports

- Imports a comprehensive set of libraries for data processing, model building, and NLP tasks:
  - Data manipulation: `fireducks.pandas`, `numpy`, `json`, `os`
  - PyTorch and utilities: `torch`, `torch.nn`, `torch.utils.data`, `torch.optim`
  - Tokenization: `nltk`
  - Progress and visualization: `tqdm`, `matplotlib`
  - Miscellaneous: `collections`, `linecache`, `random`, etc.

### 3. Preprocessing and Regular Expressions

- **Regular Expression Patterns:**  
  Patterns are compiled to clean and normalize the lyrics:
  ```python
  RE_BRACKETS = re.compile(r'\[.*?\]')
  RE_NEWLINE = re.compile(r'\n')
  RE_MULTISPACE = re.compile(r'\s+')
  RE_SPECIAL = re.compile(r'[^a-z0-9\s_]')
  RE_EDGE_NEWLINE = re.compile(r'^[\s_newline_\s]*|[_newline_]*$')
  ```
- **Text Preprocessing Function:**  
  Removes section titles, replaces newlines, removes special characters, and trims edge whitespace.

### 4. Data Extraction

- **Filtering and Preprocessing:**  
  The CSV is iteratively loaded in chunks, filtering for English (`language_ft == 'en'`) and pop genre (`tag == 'pop'`). Each lyric is preprocessed and written to a new file, `english_lyrics.txt`.

### 5. Vocabulary Construction

- **Tokenization & Vocab Building:**  
  A function tokenizes lyrics lines and builds a vocabulary, including only words that meet a minimum frequency threshold. Special tokens `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>` are included.
- **Vocab Storage:**  
  The vocabulary is saved to and loaded from a JSON file.

### 6. Dataset and DataLoader

- **LyricsDataset Class:**  
  A custom PyTorch `Dataset` that:
  - Efficiently indexes songs and lines.
  - Supports multiple training strategies (`sliding_window`, `full_context`, `verse_chorus`).
  - Prepares input/target pairs by tokenizing and padding/truncating sequences.
- **DataLoader Function:**  
  Prepares batches using the custom dataset for model training.

### 7. Model Building

- **Encoder:**  
  - Bidirectional LSTM encoder with embedding and dropout layers.
  - Returns processed hidden and cell states.

- **Attention:**  
  - Attention mechanism computes context vectors for the decoder, enabling the model to focus on relevant input tokens.

---

## How it Works (Notebook Code Highlights)

- **Data Preparation:**  
  The notebook first ensures all data is downloaded, filtered, and normalized for NLP processing.
- **Vocabulary:**  
  The vocabulary is built dynamically based on word frequency in the dataset.
- **Dataset Construction:**  
  The dataset class enables flexible sampling strategies, including context windows and verse/chorus detection.
- **Model:**  
  The encoder and (partially shown) attention modules are designed for sequence modeling, suitable for lyrics generation tasks.

---

## Usage

This notebook is intended for execution in a Kaggle, Colab, or similar Python environment with access to the required libraries and sufficient memory for handling large lyric datasets.

**Steps:**
1. Run the notebook cells sequentially to download, preprocess, and process the dataset.
2. Build the vocabulary and save it for downstream tasks.
3. Initialize the dataset and data loader for training.
4. Build and train the seq2seq model.

---

## Notes

- The notebook extensively uses Python comments and markdown to explain each processing step.
- All logic is contained within the notebook; no external scripts are necessary.
- Only code and procedures present in the notebook are described here—no extra steps or assumptions are documented.

---
