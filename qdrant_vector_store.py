import logging
import sys
import uuid;   
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document)
from llama_index.core import StorageContext, ServiceContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore;
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import fastembed;
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


global query_engine
query_engine = None

global index
index = None

def init_llm():
    model = ChatOllama(model="llama3.2")
    #embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")
    embed_model=fastembed.FastEmbedEmbeddings()
    Settings.llm = model
    Settings.embed_model = embed_model


def init_index():
    global index

    # read documents in docs directory
    # the directory contains data set related to red team and blue team cyber security strategy
    reader = SimpleDirectoryReader(input_dir="C:/Users/Dell/Downloads/sample", recursive=True)
    documents = reader.load_data()

    logging.info("index creating with `%d` documents", len(documents))

    # create large document with documents for better text balancing
    document = Document(text="\n\n".join([doc.text for doc in documents]),id_=str(uuid.uuid4()))

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
   # qdrant_client.delete_collection("rag1")

    # qdrant vector store with enabling hybrid search
    vector_store = QdrantVectorStore(
        collection_name="rag1",
        client=qdrant_client,
        enable_hybrid=True,
        batch_size=20
    )

    # storage context and service context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        node_parser=node_parser,
    )

    # initialize vector store index with qdrant
    index = VectorStoreIndex.from_documents(
        [document],
        service_context=service_context,
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )


def init_query_engine():
    global query_engine
    global index

    # after retrieval, we need to replace the sentence with the entire window from the metadata by defining a
    # MetadataReplacementPostProcessor and using it in the list of node_postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # re-ranker with BAAI/bge-reranker-base model
    rerank = SentenceTransformerRerank(
        top_n=2,
        model="BAAI/bge-reranker-base"
    )

    # similarity_top_k configure the retriever to return the top 3 most similar documents, the default value of similarity_top_k is 2
    # use meta data post processor and re-ranker as post processors
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        node_postprocessors=[postproc, rerank],
    )


def chat(input_question, user):
    global query_engine

    response = query_engine.query(input_question)
    logging.info("response from llm - %s", response)

    # view sentece window retrieval window and origianl text
    logging.info("sentence window retrieval window - %s", response.source_nodes[0].node.metadata["window"])
    logging.info("sentence window retrieval orginal_text - %s", response.source_nodes[0].node.metadata["original_text"])

    return response.response