from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from llama_index.core import  SimpleDirectoryReader
from qdrant_client import QdrantClient
import uuid;

from llama_index.core import (Settings, VectorStoreIndex,  Document)
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser

from langchain_community.embeddings import OllamaEmbeddings
from llama_index.vector_stores.qdrant import QdrantVectorStore;

class ChunkVectorStore:

  def __init__(self) -> None:
    pass

  def store_to_vector_database(self, chunks):
    
   
    # documents = chunks
    
    # print("index creating with %d documents", len(documents))
    # print("Metadata of [0]", documents[0])
    reader = SimpleDirectoryReader(input_dir="C:/Users/Dell/Downloads/sample", recursive=True)
    documents = reader.load_data()
    # create large document with documents for better text balancing
    document = Document(text="\n\n".join([doc.text for doc in documents]),id_=str(uuid.uuid4()),get_doc_id=False)
    # sentece window node parser
    # window_size = 3, the resulting window will be three sentences long
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    
    # create qdrant client
    qdrant_client = QdrantClient(f"https://2ed75e54-a92a-4c25-9ba0-4cb1c04f22ee.us-west-2-0.aws.cloud.qdrant.io:6333",api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQzMDc2NzMwfQ.wq2s3ncXAlmDL4GcdWXuB4BNbxmuynFki-177vn8mV0")

    # delete collection if exists,
    # in production application, the collection needs to be handle without deleting
  
    # qdrant vector store with enabling hybrid search
    vector_store = QdrantVectorStore(
        collection_name="rag1",
        client=qdrant_client,
        enable_hybrid=False,
        batch_size=20,
        fastembed_sparse_model=None,
        llm=None
    )
    ollama_emb = OllamaEmbeddings(
          model="llama3.2",
      )
    # storage context and service context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # service_context = ServiceContext.from_defaults(
    #     llm=None,
    #     embed_model=ollama_emb,
    #     node_parser=node_parser,
    # )
  
  
    return VectorStoreIndex.from_documents(
        [document],
        storage_context=storage_context,
        embed_model=ollama_emb
    )
    
  
  def split_into_chunks(self, file_path: str):
    #doc = PyPDFLoader(file_path).load()
    documents = SimpleDirectoryReader("C:/Users/Dell/Downloads/sample").load_data()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    # chunks = text_splitter.split_documents(doc)
    # chunks = filter_complex_metadata(chunks)
    # for chunk in chunks:
    #     chunk.metadata["get_doc_id"] = uuid.uuid4()
    return documents