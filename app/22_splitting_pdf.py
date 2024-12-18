from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
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
    pages = loader.load_and_split(text_splitter)
    print(pages[5])
    print(len(pages))

if __name__ == "__main__":
    main()
