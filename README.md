# BASE-Core-AI-LLM — first MIRAI prototype

> **This prototype has been superseded by [Skyresist/MIRAI](https://github.com/Skyresist/MIRAI)**, the full offline voice assistant now running on the robot's NVIDIA Jetson kiosk.

The first text-only version of MIRAI, BINUS ASO School of Engineering's campus assistant:

- Loads university FAQ PDFs and splits them into question–answer blocks, so a question stays together with its answer
- Retrieves with multilingual sentence embeddings plus a keyword bonus for campus terms
- Expands queries with campus vocabulary (e.g. *UKM* → student clubs)
- Answers through a local LLM (Ollama) in Bahasa Indonesia or English, constrained to the retrieved context

Run: put PDFs in `data/pdfs/`, then `python app.py`.
