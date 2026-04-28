import ollama

def detect_language(text):
    indo_words = [
        "apa", "bagaimana", "dimana", "di mana", "kapan", "kenapa",
        "siapa", "fasilitas", "kampus", "mahasiswa", "jurusan",
        "akreditasi", "kuliah", "biaya", "pendaftaran"
    ]

    text_lower = text.lower()

    for word in indo_words:
        if word in text_lower:
            return "Bahasa Indonesia"

    return "English"

def ask_llm(question, context):
    language = detect_language(question)

    prompt = f"""
You are a helpful and friendly university assistant chatbot for the university BINUS ASO school of engineering.

Answer language:
{language}

CRITICAL RULES:
1. Answer ONLY using facts explicitly written in the context.
2. Do NOT add examples, assumptions, explanations, or common university facilities.
3. Do NOT use outside knowledge.
4. If a list appears in the context, reproduce only the items from that list.
5. Do NOT add items that are not listed in the context.
6. If the context does not contain the answer, say exactly:
   "I'm not sure based on the available information."
7. If translating, translate only the provided context. Do not add new information.
8. Keep the answer faithful to the context, even if the answer feels incomplete.
9. Do not suggest contacting BINUS ASO unless the context says so.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        options={
        "num_predict": 900,
        "temperature": 0.0
        }
    )

    return response["message"]["content"]