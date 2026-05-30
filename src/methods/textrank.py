import re
import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextRankSummarizer:
    """Extractive summarizer using TextRank (PageRank on sentence similarity graph).

    Algorithm:
      1. Split dialogue into sentences.
      2. Build TF-IDF vectors for each sentence.
      3. Construct a cosine-similarity graph between sentences.
      4. Run PageRank to score sentences.
      5. Return top-n sentences in original order.
    """

    def __init__(self, top_n: int = 3, damping: float = 0.85, min_sim: float = 1e-4):
        self.top_n = top_n
        self.damping = damping
        self.min_sim = min_sim
        self._ensure_nltk()

    @staticmethod
    def _ensure_nltk():
        for resource in ["punkt_tab", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if resource == "punkt_tab" else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    def _split_sentences(self, text: str) -> list[str]:
        # Remove speaker tags before splitting to avoid [SP1]: polluting sentences
        cleaned = re.sub(r"\[SP\d+\]:\s*", "", text)
        cleaned = re.sub(r"#Person\d+#:\s*", "", cleaned)
        sentences = nltk.sent_tokenize(cleaned)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _build_similarity_matrix(self, sentences: list[str]) -> np.ndarray:
        if len(sentences) < 2:
            return np.ones((len(sentences), len(sentences)))

        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        try:
            tfidf = vectorizer.fit_transform(sentences)
        except ValueError:
            # All sentences are stop words — fall back to uniform similarity
            n = len(sentences)
            return np.ones((n, n)) / n

        sim = cosine_similarity(tfidf, tfidf)
        # Zero out self-similarity and values below threshold
        np.fill_diagonal(sim, 0)
        sim[sim < self.min_sim] = 0
        return sim

    def summarize(self, dialogue: str) -> str:
        sentences = self._split_sentences(dialogue)

        if len(sentences) == 0:
            return dialogue
        if len(sentences) <= self.top_n:
            return " ".join(sentences)

        sim_matrix = self._build_similarity_matrix(sentences)
        graph = nx.from_numpy_array(sim_matrix)

        try:
            scores = nx.pagerank(graph, alpha=self.damping, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            # Fall back to degree centrality if PageRank fails to converge
            scores = nx.degree_centrality(graph)

        ranked = sorted(scores, key=scores.get, reverse=True)[: self.top_n]
        selected = sorted(ranked)  # preserve original order
        return " ".join(sentences[i] for i in selected)

    def summarize_batch(self, dialogues: list[str]) -> list[str]:
        return [self.summarize(d) for d in dialogues]
