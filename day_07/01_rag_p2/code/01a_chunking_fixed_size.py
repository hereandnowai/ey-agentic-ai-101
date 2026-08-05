from docs_utils import load_documents, print_header

def fixed_size_chunks(text, chunk_size=300):
    chunks = []
    for start in range(0, len(text), chunk_size):
        end = start + chunk_size
        chunk = text[start:end]
        chunk = chunk.strip()
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    documents = load_documents()
    doc = None
    for d in documents:
        if d["source"] == "product_features.md":
            doc = d
            break
    assert doc is not None, "product_features.md not found in data/documents"

    chunks = fixed_size_chunks(doc["text"])
    header = f"Fixed-size chunking -> {len(chunks)} chunks from {doc['source']}"
    print_header(header)

    preview_chunks = chunks[:3]
    for i, c in enumerate(preview_chunks):
        chunk_length = len(c)
        print(f"--- chunk {i} ({chunk_length} chars) ---\n{c}\n")
