# memory.py
import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class HybridMemory:
    def __init__(self, json_path="daniel_memory.json", chroma_path="chroma_store"):
        self.json_path = json_path
        self.chroma_path = chroma_path

        # === Short-term memory (chat turns) ===
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                self.chat_history = json.load(f)
        else:
            self.chat_history = []

        # === Long-term memory (semantic facts) ===
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.chroma = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )

    # ---------------- SHORT-TERM ----------------
    def save_turn(self, role, content):
        """Save a single turn of chat history."""
        self.chat_history.append({"role": role, "content": content})
        with open(self.json_path, "w") as f:
            json.dump(self.chat_history, f, indent=2)

    def load_history(self, limit=10):
        """Get the last few conversation turns."""
        return self.chat_history[-limit:]

    # ---------------- LONG-TERM ----------------
    def save_fact(self, fact):
        """Store a long-term fact in vector DB."""
        self.chroma.add_texts([fact])
        self.chroma.persist()

    def recall(self, query, k=3):
        """Retrieve relevant facts from vector DB."""
        docs = self.chroma.similarity_search(query, k=k)
        return [d.page_content for d in docs]
