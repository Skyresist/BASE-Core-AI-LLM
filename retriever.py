from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


class Retriever:
    def __init__(self, documents):
        self.chunks = []
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.embeddings = None
        self._process_docs(documents)

    def _split_into_faq_blocks(self, text, source):
        """
        Tries to keep FAQ questions and answers together.
        Better than cutting every 1500 characters randomly.
        """

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = []
        current_block = []

        question_patterns = [
            r".*\?$",
            r"^(Q|Question|Pertanyaan)\s*[:\-]",
            r"^Apa\s+",
            r"^Bagaimana\s+",
            r"^Dimana\s+",
            r"^Di mana\s+",
            r"^Kapan\s+",
            r"^Siapa\s+",
            r"^Mengapa\s+",
            r"^Kenapa\s+",
            r"^Can\s+",
            r"^What\s+",
            r"^How\s+",
            r"^Where\s+",
            r"^When\s+",
            r"^Who\s+",
            r"^Why\s+",
        ]

        def is_question(line):
            return any(re.match(pattern, line, re.IGNORECASE) for pattern in question_patterns)

        for line in lines:
            if is_question(line) and current_block:
                blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        cleaned_blocks = []
        for block in blocks:
            if len(block.strip()) > 40:
                cleaned_blocks.append({
                    "text": block.strip(),
                    "source": source
                })

        return cleaned_blocks

    def _fallback_chunk_text(self, text, source, chunk_size=1800, overlap=300):
        chunks = []
        start = 0

        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": source
                })

            start += chunk_size - overlap

        return chunks

    def _process_docs(self, documents):
        for doc in documents:
            faq_blocks = self._split_into_faq_blocks(doc["text"], doc["source"])

            if len(faq_blocks) >= 2:
                self.chunks.extend(faq_blocks)
            else:
                self.chunks.extend(
                    self._fallback_chunk_text(doc["text"], doc["source"])
                )

        if not self.chunks:
            return

        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

    def has_data(self):
        return self.embeddings is not None and len(self.chunks) > 0

    def _keyword_bonus(self, query, chunk_text):
        query_words = set(query.lower().split())
        chunk_words = set(chunk_text.lower().split())

        overlap = query_words.intersection(chunk_words)

        bonus = 0.0

        if overlap:
            bonus += min(len(overlap) * 0.03, 0.20)

        important_terms = [
            "ukm", "unit kegiatan mahasiswa",
            "fasilitas", "facilities",
            "akreditasi", "accreditation",
            "ban-pt", "unggul",
            "d'base", "aerobase", "kendo",
            "library", "canteen",
            "lab", "laboratorium",
            "expression technique", "physics", "3d printer",
            "monozukuri", "ergonomic", "computer"
        ]

        query_lower = query.lower()
        chunk_lower = chunk_text.lower()

        for term in important_terms:
            if term in query_lower and term in chunk_lower:
                bonus += 0.15

        return bonus

    def retrieve(self, query, top_k=5):
        if not self.has_data():
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_embedding, self.embeddings).flatten()

        scored_results = []

        for idx, sim in enumerate(sims):
            chunk = self.chunks[idx]
            bonus = self._keyword_bonus(query, chunk["text"])
            final_score = float(sim) + bonus

            scored_results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": final_score,
                "embedding_score": float(sim),
                "keyword_bonus": bonus
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)

        return scored_results[:top_k]