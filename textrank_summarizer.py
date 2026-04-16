"""
TextRank Summarizer Module
Provides extractive summarization using TextRank algorithm.
"""

import re
import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources once (if not already present)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextRankSummarizer:
    """Extractive summarizer based on TextRank algorithm."""

    def __init__(self, stop_words='english', lemmatize=True, top_n=3, min_sentence_len=5):
        """
        Args:
            stop_words: language for stopwords or custom list
            lemmatize: whether to apply lemmatization
            top_n: number of sentences in the summary
            min_sentence_len: minimum word count to keep a sentence
        """
        self.stop_words = set(stopwords.words(stop_words)) if isinstance(stop_words, str) else set(stop_words)
        self.lemmatize = lemmatize
        self.top_n = top_n
        self.min_sentence_len = min_sentence_len
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None

    def _clean_text(self, text):
        """Keep only letters and spaces, lower case."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def _preprocess_sentence(self, sentence):
        """Tokenize, remove stopwords, optionally lemmatize."""
        words = word_tokenize(sentence)
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        if self.lemmatize:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        return ' '.join(words)

    def _get_sentences(self, article):
        """Split article into sentences and filter short ones."""
        if not isinstance(article, str):
            return []
        raw_sentences = sent_tokenize(article)
        sentences = [s for s in raw_sentences if len(word_tokenize(s)) >= self.min_sentence_len]
        return sentences

    def summarize(self, article, return_scores=False):
        """
        Generate summary for a single article.
        Returns:
            - if return_scores=False: summary string
            - if return_scores=True: (summary_string, scores_dict)
        """
        sentences = self._get_sentences(article)
        if not sentences:
            return "" if not return_scores else ("", {})
        if len(sentences) <= self.top_n:
            summary = " ".join(sentences)
            scores = {i: 1.0 for i in range(len(sentences))}
            return (summary, scores) if return_scores else summary

        # Preprocess sentences
        preprocessed = [self._preprocess_sentence(self._clean_text(s)) for s in sentences]

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(preprocessed)
        except ValueError:
            # All sentences became empty after preprocessing
            summary = " ".join(sentences[:self.top_n])
            return (summary, {}) if return_scores else summary

        # Cosine similarity matrix
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0)

        # Build graph and run PageRank
        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph)

        # Select top_n sentences preserving original order
        ranked = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
        top_indices = sorted([idx for _, idx in ranked[:self.top_n]])
        summary = " ".join([sentences[i] for i in top_indices])

        return (summary, scores) if return_scores else summary

    def evaluate_dataset(self, data, article_col='article', ref_col='highlights', verbose=False):
        """
        Evaluate on a whole dataset (pandas DataFrame).
        Returns average ROUGE scores and detailed results.
        """
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        results = []
        total_rouge1 = {'precision': 0, 'recall': 0, 'fmeasure': 0}
        total_rouge2 = {'precision': 0, 'recall': 0, 'fmeasure': 0}
        total_rougeL = {'precision': 0, 'recall': 0, 'fmeasure': 0}

        iterator = data.iterrows()
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(data), desc="Evaluating")

        for idx, row in iterator:
            article = row[article_col]
            reference = row[ref_col]
            summary = self.summarize(article)
            if not summary:
                continue
            scores = scorer.score(reference, summary)

            results.append({
                'index': idx,
                'reference': reference,
                'generated': summary,
                'rouge1': scores['rouge1']._asdict(),
                'rouge2': scores['rouge2']._asdict(),
                'rougeL': scores['rougeL']._asdict()
            })

            total_rouge1['precision'] += scores['rouge1'].precision
            total_rouge1['recall'] += scores['rouge1'].recall
            total_rouge1['fmeasure'] += scores['rouge1'].fmeasure

            total_rouge2['precision'] += scores['rouge2'].precision
            total_rouge2['recall'] += scores['rouge2'].recall
            total_rouge2['fmeasure'] += scores['rouge2'].fmeasure

            total_rougeL['precision'] += scores['rougeL'].precision
            total_rougeL['recall'] += scores['rougeL'].recall
            total_rougeL['fmeasure'] += scores['rougeL'].fmeasure

        n = len(results)
        avg = {
            'rouge1': {k: v/n for k, v in total_rouge1.items()},
            'rouge2': {k: v/n for k, v in total_rouge2.items()},
            'rougeL': {k: v/n for k, v in total_rougeL.items()}
        }
        return avg, results