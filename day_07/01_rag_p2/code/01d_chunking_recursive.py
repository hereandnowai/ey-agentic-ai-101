from docs_utils import load_documents, print_header, recursive_chunks

if __name__ == "__main__":
    documents = load_documents()
    doc = None
    for d in documents:
        if d["source"] == "product_features.md":
            doc = d
            break
    assert doc is not None, "product_features.md not found in data/documents"

    chunks = recursive_chunks(doc["text"], chunk_size=300)
    header = f"Recursive chunking -> {len(chunks)} chunks from {doc['source']}"
    print_header(header)

    preview_chunks = chunks[:2]
    for i, c in enumerate(preview_chunks):
        chunk_length = len(c)
        print(f"--- chunk {i} ({chunk_length} chars) ---\n{c}\n")
