from langchain_community.document_loaders import PyPDFLoader

def main():
    file_path = (
        "./fixtures/nais.pdf"
    )
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    print(pages[len(pages)-1])
    print(len(pages))

if __name__ == "__main__":
    main()
