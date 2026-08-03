# We rarely embed a whole document. We split it into CHUNKS and embed each.


def chunk_by_chars(text: str, size: int, overlap: int=0) -> list[str]:
    """Split text into ~size-char chunks with optional overlap (keeps context across cuts)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]

policy = (
    "Home loan interest rates are at an all-time low, making it a great time to buy a house. "
    "However, it's important to consider your financial situation and long-term plans before making a decision"
    "The maximum tenure is 30 years, and the minimum tenure is 5 years. "
    "A processing fee of 1% of the loan amount is applicable, and prepayment charges may apply if you pay off the loan early. "
)

print("=== big chunks (200 chars) ===")
for i, c in enumerate(chunk_by_chars(policy, 200)):
    print(f" chunk {i} ({len(c)} chars): {c}")

print("\n=== small chunks (90 chars, 20 overlap) ===")
for i, c in enumerate(chunk_by_chars(policy, 90, overlap=20)):
    print(f" chunk {i} ({len(c)} chars): {c}")