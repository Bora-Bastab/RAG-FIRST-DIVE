from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from embedding_utils import get_embedding_model
import os
import asyncio

app = Flask(__name__)

# Define the path to the FAISS index
INDEX_DIR = "travel_knowledge_base"

# Define the prompt template for RAG
RAG_PROMPT_TEMPLATE = """
You are Agent G, an expert at the Iceland Travel Agency. Use the following context to craft a concise, human-like response to the traveler's query. Focus on providing a tailored, engaging, and accurate answer.

Context:
{context}

---

Traveler's Query: {question}
"""

# In-memory chat history (for demonstration purposes)
chat_history = []

def is_iceland_query(query: str) -> bool:
    """
    Determines if the query is related to Iceland and should use RAG.
    """
    iceland_keywords = [
        "Iceland", "Rauðfeldsgjá Gorge", "Vatnshellir Cave", "Blue Lagoon", "Golden Circle", "Fjaðrárgljúfur Canyon",
        "Jökulsárlón", "Reynisfjara", "Skógafoss", "Seljalandsfoss", "Northern Lights", "Aurora Borealis"
    ]
    query_lower = query.lower()
    print(f"Checking query '{query}' against Iceland keywords: {iceland_keywords}")  # Debugging: Log the query and keywords
    return any(keyword.lower() in query_lower for keyword in iceland_keywords)

async def generate_travel_response(query: str, history: list = None) -> str:
    """
    Generates a response using the Mistral model via Ollama.
    """
    llm = OllamaLLM(model="mistral")
    prompt = "\n".join(history) + f"\nTraveler: {query}" if history else query
    return llm.invoke(prompt)

async def process_travel_query(query: str, history: list = None):
    """
    Handles travel queries using either direct Mistral or RAG-enhanced responses.
    """
    print(f"Processing query: '{query}'")  # Debugging: Log the query

    # Check if the query is travel-related
    is_iceland = is_iceland_query(query)
    print(f"Is Iceland query: {is_iceland}")  # Debugging: Log the classification

    if not is_iceland:
        response = await generate_travel_response(query, history)
        return {"response": response, "sources": None}

    # Check if the FAISS index exists
    if not os.path.exists(INDEX_DIR):
        print(f"FAISS index not found at {INDEX_DIR}. Falling back to Mistral.")  # Debugging: Log missing index
        return {"response": "The travel knowledge base is not available. Please try again later.", "sources": None}

    # Load the FAISS index
    embedding_model = get_embedding_model()
    travel_knowledge_base = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)

    # Retrieve the most relevant document (top 1)
    search_results = travel_knowledge_base.similarity_search_with_score(query, k=1)

    if not search_results:
        print("No relevant information found in FAISS index.")  # Debugging: Log no results
        return {"response": "I couldn't find any relevant information. Please try another query.", "sources": None}

    # Prepare the context for RAG
    doc, score = search_results[0]
    context = doc.page_content
    source = doc.metadata.get("source", "Unknown")

    print("Retrieved Travel Insight:")
    print(f"Content: {context}\nSource: {source}\nScore: {score}\n")  # Debugging: Log the retrieved document

    # Generate the response using Mistral
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(context=context, question=query)
    llm = OllamaLLM(model="mistral")
    response = llm.invoke(prompt)

    # Return the response with the source
    return {"response": response, "sources": [source]}

@app.route("/")
def home():
    """
    Renders the home page.
    """
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    """
    Handles the query from the web interface.
    """
    data = request.json
    query_text = data.get("query", "").strip()
    print(f"Received query from UI: '{query_text}'")  # Debugging: Log the query

    if not query_text:
        return jsonify({"response": "Please enter a valid query.", "sources": None})

    # Process the query
    result = asyncio.run(process_travel_query(query_text))
    print(f"Query result: {result}")  # Debugging: Log the result

    # Add the query and response to the chat history
    chat_history.append({"query": query_text, "response": result["response"], "sources": result["sources"]})

    return jsonify(result)

@app.route("/history", methods=["GET"])
def history():
    """
    Returns the chat history.
    """
    return jsonify(chat_history)

if __name__ == "__main__":
    app.run(debug=True)
