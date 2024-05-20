from config import hf_embeddings

def main():
    embedding = hf_embeddings.embed_query("That is a happy dog")

    print(embedding)

    print(len(embedding))


if __name__ == "__main__":
    main()
