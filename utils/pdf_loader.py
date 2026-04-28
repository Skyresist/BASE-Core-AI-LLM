from pypdf import PdfReader
import os

def load_pdfs(folder_path):
    documents = []

    if not os.path.exists(folder_path):
        print(f"Warning: folder not found -> {folder_path}")
        return documents

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder_path, file_name)

        try:
            reader = PdfReader(path)
            text = ""

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Warning: failed reading page {page_num} of {file_name}: {e}")

            text = text.strip()

            print(f"Loaded {file_name} | extracted chars: {len(text)}")

            if text:
                documents.append({
                    "source": file_name,
                    "text": text
                })
            else:
                print(f"Warning: {file_name} produced no extractable text")

        except Exception as e:
            print(f"Warning: failed to load {file_name}: {e}")

    print(f"Total usable documents: {len(documents)}")
    return documents