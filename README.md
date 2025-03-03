# Multi-AI-Agent System with RAG

This project implements a multi-agent system using Retrieval-Augmented Generation (RAG) for handling various types of customer queries. The system uses FAISS for efficient vector similarity search and the Llama 3.3 model via Groq's API for query classification and response generation.

## System Architecture

The system consists of several specialized agents that handle different types of queries:

1. **Billing Agent** - Handles billing-related queries (invoices, payments, refunds)
2. **Technical Agent** - Handles technical support queries (product functionality, errors)
3. **Order Agent** - Handles order-related queries (order status, shipping, returns)

Each agent has its own knowledge base stored in a FAISS vector database, which enables fast and efficient retrieval of relevant information.

## Key Components

- **Query Router**: Classifies incoming queries and routes them to the appropriate specialized agent
- **RAG-based Agents**: Domain-specific agents that combine retrieval and generation to provide accurate responses
- **FAISS Vector Database**: Stores and retrieves document embeddings for efficient similarity search
- **Data Processor**: Handles data ingestion, preprocessing, and vectorization
- **API Interface**: FastAPI-based REST API for interacting with the system

## Getting Started

### Prerequisites

- Python 3.8+
- FAISS
- Sentence Transformers
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-ai-agent-system.git
cd multi-ai-agent-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key"
```

### Data Ingestion

Before running the system, you need to ingest and process the data for each domain:

```bash
python data/ingest_script.py --domain all
```

This will:
- Load the raw data from CSV files
- Chunk and preprocess the text
- Generate embeddings
- Create FAISS indices for each domain

### Running the API Server

Start the API server with:

```bash
python src/main.py
```

The API will be available at `http://localhost:8000`.

## API Usage

### Process a Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I update my billing information?"}'
```

### Health Check

```bash
curl http://localhost:8000/api/health
```

## Configuration

The system configuration is stored in YAML files:

- `config/app_config.yaml`: Main application configuration
- `config/logging_config.yaml`: Logging configuration
- `data/data_config.yaml`: Data ingestion configuration

## Extending the System

### Adding a New Agent

1. Create a new agent class in `src/agents/`
2. Add relevant data sources in `data/data_config.yaml`
3. Update the QueryRouter in `src/core/query_router.py` to include the new agent

### Adding New Data Sources

1. Add the new data source configuration in `data/data_config.yaml`
2. Run the ingestion script with the appropriate domain

## License

This project is licensed under the MIT License - see the LICENSE file for details.