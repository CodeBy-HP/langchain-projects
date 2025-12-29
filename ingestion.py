import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()


if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt", encoding="UTF-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
    )

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.getenv("INDEX_NAME")
    )
    print("finish")
