import os
import warnings
from contextlib import redirect_stdout, redirect_stderr

from secret import patch_response
from utils.pdf_loader import load_pdfs
from retriever import Retriever
from llm import ask_llm


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEBUG_RETRIEVAL = True


def build_context(results):
    if not results:
        return ""

    context = ""
    for r in results:
        context += f"[Source: {r['source']} | Score: {r['score']:.2f}]\n{r['text']}\n\n"
    return context

def expand_query(query):
    q = query.lower()
    expanded = query

    if "ukm" in q:
        expanded += " Unit Kegiatan Mahasiswa organisasi mahasiswa klub club komunitas D'BASE AeroBASE Himpunan Mahasiswa Kendo BASE CORE"

    if "unit kegiatan mahasiswa" in q:
        expanded += " UKM organisasi mahasiswa klub club D'BASE AeroBASE Himpunan Mahasiswa Kendo BASE CORE"

    if "fasilitas" in q or "facilities" in q:
        expanded += " fasilitas BINUS ASO laboratorium lab Expression Technique Lab Physics Lab 3D Printer Lab Monozukuri Lab Ergonomic Lab Computer Lab Library Canteen"

    if "akreditasi" in q or "accreditation" in q:
        expanded += " akreditasi Unggul BAN-PT SK Automotive and Robotics Engineering Product Design Engineering Business Engineering Computer Engineering Industrial Engineering"
    
    if "jurusan" in q or "major" in q:
        expanded += "Automotive & Robotics Engineering Product Design Engineering Business Engineering"

    return expanded

def main():
    # Silent loading
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            docs = load_pdfs("data/pdfs")
            retriever = Retriever(docs)

    conversation_history = []

    while True:
        q = input("You: ").strip()

        if not q:
            print("Bot: Please enter a question.\n")
            continue

        if q.lower() == "reset":
            conversation_history.clear()
            print("Bot: Conversation memory has been reset.\n")
            continue

        if q.lower() in ["exit", "end chat", "bye"]:
            print("Bot: Goodbye.\n")
            break

        if "who is the number one idol" in q.lower():
            print("Bot:", patch_response("K4N_7A"), "\n")
            continue

        # Retrieval
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                query_for_retrieval = expand_query(q)
                results = retriever.retrieve(query_for_retrieval)

        context = build_context(results)

        if not context:
            print("Bot: I'm not sure based on the available information.\n")
            continue

        if DEBUG_RETRIEVAL:
            print("\n--- Retrieved Context Preview ---")
            print(context[:2500])
            print("---------------------------------\n")

        answer = ask_llm(q, context)
        answer = patch_response(answer)

        conversation_history.append({"user": q, "bot": answer})

        print("Bot:", answer, "\n")


if __name__ == "__main__":
    main()