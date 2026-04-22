import os
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END


# -----------------------------
# State definition
# -----------------------------
class GraphState(TypedDict, total=False):
    query: str
    response: str
    confidence: float
    context: str


# -----------------------------
# Load PDF
# -----------------------------
pdf_path = "data/HLD.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

if not docs:
    raise ValueError("No content could be loaded from the PDF.")


# -----------------------------
# Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

if not chunks:
    raise ValueError("No chunks were created from the PDF.")


# -----------------------------
# Create embeddings + vector DB
# -----------------------------
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="db"
)

retriever = db.as_retriever(search_kwargs={"k": 4})


# -----------------------------
# LLM (local via Ollama)
# -----------------------------
llm = OllamaLLM(model="llama3.2", temperature=0)


# -----------------------------
# Nodes
# -----------------------------
def process(state: GraphState) -> GraphState:
    query = state["query"]

    try:
        retrieved_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        if not context.strip():
            state["response"] = "I could not find relevant information in the document."
            state["confidence"] = 0.3
            state["context"] = ""
            return state

        prompt = f"""
You are a helpful assistant answering questions only from the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Answer only from the context.
- If the answer is not in the context, say: "The answer is not available in the document."
- Keep the answer clear and concise.
"""

        ai_response = llm.invoke(prompt)

        state["response"] = ai_response
        state["confidence"] = 0.8
        state["context"] = context
        return state

    except Exception as e:
        state["response"] = f"Error during processing: {str(e)}"
        state["confidence"] = 0.0
        state["context"] = ""
        return state


def output(state: GraphState) -> GraphState:
    print("\nAnswer:", state.get("response", "No response generated."))
    return state


def hitl(state: GraphState) -> GraphState:
    print("\nEscalated to human.")
    state["response"] = "Human will respond."
    print("Answer:", state["response"])
    return state


def route(state: GraphState) -> str:
    if state.get("confidence", 0) < 0.5:
        return "hitl"
    return "output"


# -----------------------------
# Build graph
# -----------------------------
graph = StateGraph(GraphState)

graph.add_node("process", process)
graph.add_node("output", output)
graph.add_node("hitl", hitl)

graph.set_entry_point("process")

graph.add_conditional_edges(
    "process",
    route,
    {
        "output": "output",
        "hitl": "hitl",
    }
)

graph.add_edge("output", END)
graph.add_edge("hitl", END)

app = graph.compile()


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    query = input("Ask: ").strip()

    if not query:
        print("Please enter a valid query.")
    else:
        app.invoke({"query": query})