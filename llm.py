import ollama


def detect_language(text):
    indo_words = [
        "apa", "bagaimana", "dimana", "di mana", "kapan", "kenapa",
        "siapa", "fasilitas", "kampus", "mahasiswa", "jurusan",
        "akreditasi", "kuliah", "biaya", "pendaftaran", "ukm",
        "unit kegiatan mahasiswa", "ada", "aja"
    ]

    text_lower = text.lower()

    for word in indo_words:
        if word in text_lower:
            return "Bahasa Indonesia"

    return "English"


def ask_llm(question, context):
    language = detect_language(question)

    prompt = f"""
You are a strict university FAQ assistant for BINUS ASO.

Answer language:
{language}

CRITICAL RULES:
1. Answer ONLY using facts explicitly written in the context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT add examples, assumptions, or common university information.
5. If a list appears in the context, reproduce only the items from that list.
6. Do NOT add list items that are not written in the context.
7. If the context contains names, dates, SK numbers, program names, facilities, or organization names, preserve them accurately.
8. If the answer is not clearly found in the context, say exactly:
   "I'm not sure based on the available information."
9. If translating, translate only the information from the context.
10. Keep the answer clear and structured.
11. Do not repeat the same list or item twice.
12. Do not suggest contacting BINUS ASO unless the context says so.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "num_predict": 900
        }
    )

    return response["message"]["content"]