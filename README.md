# RAG-Based Travel Assistant

Welcome to the Iceland Travel Agency's AI Assistant! This project uses **Retrieval-Augmented Generation (RAG)** to provide tailored travel recommendations for Iceland.

## Features
- **RAG Workflow**: Retrieves relevant information from a FAISS index and generates responses using Mistral.
- **Interactive UI**: Built with Flask and HTML/CSS for a seamless user experience.
- **Scalable**: Can be run locally or deployed to the cloud.

## Prerequisites

Before running the project, ensure you have the following installed:

1. **Python 3.11+**: Download and install Python from [python.org](https://www.python.org/).
2. **Ollama**: Install Ollama to run Mistral locally. Follow the instructions at [ollama.com](https://ollama.com).
3. **Git**: Install Git from [git-scm.com](https://git-scm.com/).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bora-Bastab/First_dive_into_RAG.git
   cd First_dive_into_RAG

   
## Running the Project

**Terminal Mode & UI Mode**
Run the interactive travel assistant in the terminal:

```bash
   python query_prompting.py

   Enter your query (e.g., "Find hidden spots for couples in Iceland (winter)").
   View the response and source of information.

Start the Flask server:

```bash
   python app.py
   Open your browser and navigate to http://127.0.0.1:5000.
   Enter your query in the UI and view the response.
