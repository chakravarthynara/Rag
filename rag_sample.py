import os
import glob
import tiktoken
import numpy as np
from IPython.display import Markdown, display
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sklearn.manifold import TSNE
import plotly.graph_objects as go

import truststore
truststore.inject_into_ssl()



MODEL = "gpt-4o-mini"
db_name = "vector_db"
load_dotenv(override=True)


knowledge_base_path = "knowledge-base/**/*.md"
files = glob.glob(knowledge_base_path, recursive=True)
print(f"Found {len(files)} files in the knowledge base")

entire_knowledge_base = ""

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        entire_knowledge_base += f.read()
        entire_knowledge_base += "\n\n"

print(f"Total characters in knowledge base: {len(entire_knowledge_base):,}")

encoding = tiktoken.encoding_for_model(MODEL)
tokens = encoding.encode(entire_knowledge_base)
token_count = len(tokens)
print(f"Total tokens for {MODEL}: {token_count:,}")

import truststore
truststore.inject_into_ssl()  # <-- add this at the very top

from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

tokenizer = model.tokenizer
tokens = tokenizer.encode(entire_knowledge_base, add_special_tokens=False)
print("Total tokens:", len(tokens))


# # Count tokens using the model's tokenizer
# # SentenceTransformers wraps Hugging Face tokenizers internally
# tokenizer = model.tokenizer

# # Example text
# text = entire_knowledge_base  # your full knowledge base string

# # Encode without adding special tokens
# tokens = tokenizer.encode(text, add_special_tokens=False)
# token_count = len(tokens)

# print(f"Total tokens: {token_count}")


folders = glob.glob("knowledge-base/*")

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"Loaded {len(documents)} documents")



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Divided into {len(chunks)} chunks")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_openai import OpenAIEmbeddings
API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=API_KEY,
    base_url="https://genailab.tcs.in",   # same endpoint
)

#embeddings = OpenAIEmbeddings()

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# How many documents are in the vector store? How many dimensions?

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")


result = collection.get(include=['embeddings', 'documents', 'metadatas'])

vectors = np.array(result['embeddings'])
documents = result['documents']
metadatas = result['metadatas']

# Extract document types safely
doc_types = []
for m in metadatas:
    src = m.get('source', '')
    parts = src.split('/')
    doc_type = parts[1] if len(parts) > 1 else "unknown"
    doc_types.append(doc_type)

# Map colors safely
TYPE_TO_COLOR = {
    "products": "blue",
    "employees": "green",
    "contracts": "red",
    "company": "orange",
    "unknown": "gray",
}

colors = [TYPE_TO_COLOR.get(t, "gray") for t in doc_types]

# We humans find it easier to visalize things in 2D!
# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

# Let's try 3D!

tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()


API_KEY = os.getenv("OPENAI_API_KEY")
from openai import OpenAI

import httpx
client = OpenAI(api_key=API_KEY,base_url="https://genailab.tcs.in",http_client=httpx.Client(verify=False))

# model_llm = 'azure/genailab-maas-gpt-4o' 
# llm = ChatOpenAI(temperature=0, model=model_llm,client=client)

import os
from langchain_openai import ChatOpenAI

API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="azure/genailab-maas-gpt-4o",     # your MaaS model name
    api_key=API_KEY,
    base_url="https://genailab.tcs.in",     # add /v1 here if your endpoint needs it
    temperature=0,
)



retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define our system prompt template
SYSTEM_PROMPT_TEMPLATE = """Answer the question based on the context:

Context:
{context}"""

# def answer_question(question: str) -> tuple[str, list]:
#     docs = retriever.invoke(question, k=5)
#     context = "\n\n".join(doc.page_content for doc in docs)
#     system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
#     response = llm.invoke([
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=question)
#     ])
    
#     return response.content, docs


from langchain_core.messages import SystemMessage, HumanMessage

def answer_question(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])

    return response.content, docs


query = "Please explain what Insurellm is in a couple of sentences"
answer, context_docs = answer_question(query)

print(f"Question: {query}\n")
print(f"Answer: {answer}\n")
print(f"Used {len(context_docs)} documents for context")
