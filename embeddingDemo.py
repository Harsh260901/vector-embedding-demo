from sentence_transformers import SentenceTransformer

def main():
    # Load a small, fast embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Input text
    text = input("Enter text to embed: ")

    # Generate embedding
    embedding = model.encode(text)

    print("\nEmbedding vector:")
    print(embedding)

    print("\nVector length:", len(embedding))


if __name__ == "__main__":
    main()
