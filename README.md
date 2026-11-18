# TF-IDF Document Similarity Checker

A small NLP learning project that takes two text documents and returns a
similarity score between 0 and 1. Every stage of the pipeline, including
TF-IDF and cosine similarity, is written from scratch in Python and NumPy so
that each step can be seen and understood.

## What this project is
A basic document similarity checker. It measures how much two texts overlap in
their important words. It is a learning project meant to show how text
similarity works internally. It is not a production tool, and it is not a
full plagiarism detector (see Limitations).

## How it works
1. **Preprocessing:** lowercase the text and remove punctuation
2. **Tokenization:** split the text into words
3. **Stopword removal:** drop very common words such as "the", "and", "from"
4. **Stemming:** reduce words to a root with a simple rule-based stemmer, so
   that forms like "learn" and "learning" match
5. **Bag of Words:** count each word to turn every document into a vector
6. **TF-IDF:** weight each word by how often it appears in the document (TF)
   and how rare it is across a collection (IDF = ln(N / df)), so distinctive
   words count more than common ones
7. **Cosine similarity:** compare the two vectors by the angle between them.
   The result is a number from 0 (no shared weighted words) to 1 (same
   words in the same proportions)

The checker adds the two input texts to a small background collection of
documents before computing IDF. IDF only means something when a word is
measured against other documents. With just two texts, every shared word
would get a weight of zero.

## Concepts covered
Text preprocessing, tokenization, stopwords, stemming, Bag of Words, TF-IDF,
cosine similarity, the role of a background collection, and checking a
from-scratch implementation against a trusted library.

## Usage
Open the notebook in Google Colab, run the cells in order, then call:

    score = similarity(text_a, text_b)

Both inputs are plain text strings. Text read from uploaded .txt files works
the same way.

## Results
- Identical text compared with itself: 1.0
- Same topic written in different words: roughly 0.4 to 0.45
- Unrelated topics: 0.0
- No model is trained. The vocabulary and IDF values are computed from the
  documents each time.
- Verified during development against scikit-learn: the cosine function
  matched to about 1e-17. The TF-IDF scores are lower than scikit-learn's
  because scikit-learn smooths the IDF, but the ranking of documents agreed,
  apart from one near-tie.

## Limitations
- **It matches words, not meaning.** Two texts that describe the same idea with
  different vocabulary can score near 0.
- **It cannot detect opposite meanings.** Texts with nearly the same words can
  score high even if they disagree.
- **The stemmer is simple.** It only covers the patterns it has rules for and
  misses some word forms.
- **Short texts give noisy scores.** One shared word can change the result a
  lot.
- **Scores depend on the background collection.** The same two texts can
  score differently with a different collection, and the score is not a
  percentage of copied text.
- **It is not a full plagiarism checker.** It gives one overall score. It does
  not find or highlight copied passages or search external sources.
- **The tokenizer is basic.** For example, apostrophes split words into
  separate tokens.

## Future development
- Use a stronger stemmer or lemmatization
- Add word-sequence (n-gram) matching so that copied passages can be located
- Use a larger and more varied background collection
- Add a smoothed IDF option, as scikit-learn does
- Compare against embedding-based similarity, which captures meaning
- Evaluate on a labelled set of similar and dissimilar pairs, and tune a
  threshold for "similar"
- Add unit tests and a simple command-line or file-input interface
- Improve the tokenizer for punctuation and contractions

## Tech
Python and NumPy. The notebook runs in Google Colab.
