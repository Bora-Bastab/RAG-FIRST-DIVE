import os
import sys
import time
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from embedding_utils import get_embedding_model
import asyncio

# Step 1: Define the path to the FAISS index
INDEX_DIR = "travel_knowledge_base"  # Directory containing the vector index

# Step 2: Define the prompt template for RAG
RAG_PROMPT_TEMPLATE = """
You are Agent G, an expert at the Iceland Travel Agency. Use the following context to craft a concise, human-like response to the traveler's query. Focus on providing a tailored, engaging, and accurate answer.

Context:
{context}

---

Traveler's Query: {question}
"""

def is_iceland_query(query: str) -> bool:
    """
    Determines if the query is related to Iceland and should use RAG.
    """
    iceland_keywords = [
        "Iceland", "Rauðfeldsgjá Gorge", "Vatnshellir Cave", "Blue Lagoon", "Golden Circle", "Fjaðrárgljúfur Canyon",
        "Jökulsárlón", "Reynisfjara", "Skógafoss", "Seljalandsfoss", "Northern Lights", "Aurora Borealis"
    ]
    query_lower = query.lower()
    return any(keyword.lower() in query_lower for keyword in iceland_keywords)

async def generate_generic_response(query: str, history: list = None) -> str:
    """
    Generates a response using the Mistral model via Ollama for non-Iceland queries.
    """
    llm = OllamaLLM(model="mistral")
    prompt = "\n".join(history) + f"\nTraveler: {query}" if history else query
    return llm.invoke(prompt)

async def process_travel_query(query: str, history: list = None):
    """
    Handles travel queries using RAG for Iceland-related queries and Mistral for others.
    """
    # Step 3: Check if the query is about Iceland
    if not is_iceland_query(query):
        response = await generate_generic_response(query, history)
        return {"response": response, "sources": None}

    # Step 4: Check if the FAISS index exists
    if not os.path.exists(INDEX_DIR):
        return {"response": "The travel knowledge base is not available. Please try again later.", "sources": None}

    # Step 5: Load the FAISS index
    embedding_model = get_embedding_model()
    travel_knowledge_base = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)

    # Step 6: Retrieve the most relevant document (top 1)
    search_results = travel_knowledge_base.similarity_search_with_score(query, k=1)

    if not search_results:
        return {"response": "I couldn't find any relevant information. Please try another query.", "sources": None}

    # Step 7: Prepare the context for RAG
    doc, score = search_results[0]
    context = doc.page_content
    source = doc.metadata.get("source", "Unknown")

    print("Retrieved Travel Insight:")
    print(f"Content: {context}\nSource: {source}\nScore: {score}\n")

    # Step 8: Generate the response using Mistral
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(context=context, question=query)
    llm = OllamaLLM(model="mistral")
    response = llm.invoke(prompt)

    # Step 9: Return the response with the source
    return {"response": response, "sources": [source]}

def show_loader(message: str = "Thinking..."):
    """
    Displays a loader animation with a message.
    """
    chars = "/—\\|"
    for i in range(10):  # Show the loader for 10 iterations
        sys.stdout.write(f"\r{message} {chars[i % len(chars)]}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")  # Clear the loader

async def interactive_travel_assistant():
    """
    Interactive travel assistant for Agent G.
    """
    print("Welcome to the Iceland Travel Agency's AI Assistant!")
    print("Type 'exit' to quit.")
    while True:
        query = input("Traveler's Query: ")
        if query.lower() == 'exit':
            break

        # Show loader while processing the query
        show_loader("Agent G is thinking...")

        # Process the query
        result = await process_travel_query(query)

        # Display the response
        print("\rAgent G's Response:", result["response"])
        if result["sources"]:
            print("Source:", result["sources"][0])

if __name__ == "__main__":
    asyncio.run(interactive_travel_assistant())
