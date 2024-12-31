from src.helper import load_pdf_files,text_split,download_huggingface_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")


extracted_data=load_pdf_files(data='Data')
text_chunks=text_split(extracted_data)
embeddings=download_huggingface_face_embeddings()

pc = Pinecone(api_key="pcsk_4X1GkE_Ny6a48MuM5AsEWndXXRqUyT5WwxYB7T9cHosCr9fqvNNeyGFnmrFTmWxK2fGKjz")
index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


from langchain_pinecone import PineconeVectorStore

docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

