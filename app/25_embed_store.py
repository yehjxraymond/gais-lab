import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import hf_embeddings
import pandas as pd

NAIS_PICKLE = "./fixtures/nais.pkl"

def main():
    # Load from existing pickle file if it exists
    if os.path.exists(NAIS_PICKLE):
        print("Pickle file already exists. Loading DataFrame from the file.")
        df = pd.read_pickle(NAIS_PICKLE)
        print(df.head())
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    file_path = (
        "./fixtures/nais.pdf"
    )
    loader = PyPDFLoader(file_path)
    document = loader.load_and_split(text_splitter)
   
    documents_data = []

    for idx, doc in enumerate(document):
        print(f"Embedding {idx}")
        try:
            doc_embedding = hf_embeddings.embed_query(doc.page_content)

            documents_data.append({
                "id": idx,
                "content": doc.page_content,
                "source": doc.metadata,
                "embedding": doc_embedding
            })
        except Exception as e:
            print(f"Error processing document {idx}: {e}")
            # Optionally, save the partially processed DataFrame to avoid complete loss
            pd.DataFrame(documents_data).to_pickle("tmp/partial_documents_dataframe.pkl")

    # Create a DataFrame from the collected data
    df = pd.DataFrame(documents_data)

    # Save the DataFrame to a pickle file
    df.to_pickle(NAIS_PICKLE)

if __name__ == "__main__":
    main()
