Frontend :- uvicorn frontend.app:app --reload --host 0.0.0.0 --port 8000
To extract, chunk :- python data_processing\extract.py
TO embedd:- python embedding\embed_and_upsert.py
To Retrevial :- python retrevial\rag_cli.py

