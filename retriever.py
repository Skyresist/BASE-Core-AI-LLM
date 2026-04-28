from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, documents):
        self.chunks = []
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.embeddings = None
        self._process_docs(documents)

    def _chunk_text(self, text, chunk_size=2000, overlap=300):
        chunks = []
        start = 0

        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def _process_docs(self, documents):
        for doc in documents:
            chunks = self._chunk_text(doc["text"])

            for chunk in chunks:
                self.chunks.append({
                    "text": chunk,
                    "source": doc["source"]
                })

        if not self.chunks:
            return

        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

    def has_data(self):
        return self.embeddings is not None and len(self.chunks) > 0

    def retrieve(self, query, top_k=10, min_score=0.0):
        if not self.has_data():
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_embedding, self.embeddings).flatten()

        top_indices = sims.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "score": float(sims[idx])
            })

        return results