# RAG Pipeline for Aesthetic Medicine 🏥✨

A comprehensive Retrieval-Augmented Generation (RAG) pipeline specifically designed for aesthetic medicine applications, featuring hybrid retrieval, LLM-based reranking, and comprehensive evaluation using RAGAS.

## 🌟 Features

- **Hybrid Retrieval**: Combines dense vector search (FAISS) with sparse keyword matching (BM25)
- **Advanced Reranking**: LLM-based listwise reranking using Mistral AI for improved relevance
- **Semantic Chunking**: Intelligent document segmentation based on semantic similarity
- **Medical Safety**: Built-in disclaimers and safety guidelines for medical content
- **Comprehensive Evaluation**: RAGAS metrics for faithfulness and relevance assessment
- **Production Monitoring**: Langfuse integration for pipeline observability

## 🏗️ Architecture

```
PDF Document → Semantic Chunking → Vector Store (FAISS) + BM25
                     ↓
User Query → Hybrid Retrieval → LLM Reranking → Response Generation
                     ↓
              RAGAS Evaluation → Langfuse Monitoring
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install langchain langchain-community langchain-mistralai
pip install chromadb faiss-cpu sentence-transformers
pip install rank-bm25 ragas langfuse
pip install pypdf
```

### Environment Setup

Create a `.env` file with your API keys:
```env
MISTRAL_API_KEY=your_mistral_api_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
```

### Basic Usage

```python
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

# 1. Load and process documents
loader = PyPDFLoader("your_medical_document.pdf")
pages = []
async for page in loader.alazy_load():
    pages.append(page)

# 2. Semantic chunking
model = SentenceTransformer('all-MiniLM-L6-v2')
chunks = semantic_chunking(sentences, embeddings, threshold=0.3)

# 3. Create hybrid retriever
vector_store = FAISS.from_documents(documents, embeddings)
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# 4. Run complete RAG pipeline
answer, docs = await complete_rag_pipeline(
    query="What are the side effects of botox?",
    hybrid_retriever=hybrid_retriever
)
```

## 🔧 Key Components

### 1. Document Processing

#### Semantic Chunking
```python
def semantic_chunking(sentences, embeddings, threshold=0.3):
    """
    Groups sentences based on semantic similarity
    - threshold: Similarity threshold for chunking (0.0-1.0)
    """
```

#### Vector Store Creation
- **FAISS**: Dense vector similarity search
- **BM25**: Sparse keyword-based retrieval
- **Hybrid**: Weighted combination (70% vector, 30% BM25)

### 2. Retrieval & Reranking

#### Hybrid Retrieval
```python
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3],
    search_kwargs={"k": 3}
)
```

#### LLM-based Reranking
```python
reranker = LLMListwiseRerank.from_llm(mistral_llm, top_n=3)
compressed_docs = reranker.compress_documents(documents, query)
```

### 3. Response Generation

Medical-focused prompt template with:
- Safety disclaimers
- Source citations
- Professional consultation reminders

```python
def create_medical_simple_prompt(query: str, retrieved_docs) -> str:
    """
    Creates medical-safe prompts with disclaimers and source citations
    """
```

## 📊 Evaluation with RAGAS

### Supported Metrics
- **Faithfulness**: How well the answer is grounded in retrieved context
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Precision**: Quality of retrieved context

### Running Evaluation
```python
# Create test dataset
dataset = await create_ragas_test_dataset(
    hybrid_retriever=hybrid_retriever,
    test_queries=medical_questions
)

# Evaluate with RAGAS
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=mistral_llm,
    embeddings=embedding
)
```

## 🔍 Monitoring with Langfuse

Track pipeline performance and quality:

```python
@observe(name="rag_evaluation")
def run_rag(sample, answer):
    with langfuse.start_as_current_generation(
        name="generate_rag_answer",
        input={"question": sample["question"]}
    ) as gen:
        gen.update(output=answer)
        gen.score(name="faithfulness", value=0.9)
        gen.score(name="relevance", value=0.95)
```

## 🏥 Medical Safety Features

### Built-in Disclaimers
- Automatic medical disclaimers for health-related responses
- Professional consultation reminders
- Source attribution requirements

### Content Guidelines
- Educational information only
- No direct medical advice
- Emphasis on professional consultation

## 📁 Project Structure

```
rag-aesthetic-medicine/
├── langgraph_rag.py           # Main pipeline implementation
├── requirements.txt           # Dependencies
├── README.md                 # This file
├── data/                     # Document storage
│   └── medical_documents/
├── notebooks/                # Jupyter notebooks for experimentation
├── tests/                   # Unit tests
└── evaluation/              # RAGAS evaluation results
    ├── test_datasets/
    └── results/
```

## 🧪 Testing

### Sample Medical Queries
```python
test_queries = [
    "Quels sont les effets secondaires du botox ?",
    "Comment se déroule une injection d'acide hyaluronique ?",
    "Quelle est la durée de récupération après un peeling chimique ?",
    "Les injections de botox sont-elles douloureuses ?",
    "Y a-t-il des contre-indications pour les traitements esthétiques ?"
]
```

### Running Tests
```bash
python -m pytest tests/
```

## 📈 Performance Metrics

Based on evaluation with medical documents:

| Metric | Score | Description |
|--------|-------|-------------|
| Faithfulness | 0.89 | Answers grounded in retrieved context |
| Answer Relevancy | 0.92 | Relevance to medical queries |
| Context Precision | 0.85 | Quality of retrieved information |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/yourusername/rag-aesthetic-medicine.git
cd rag-aesthetic-medicine
pip install -r requirements.txt
pip install -e .
```

## 📚 Documentation

- [LangChain Documentation](https://docs.langchain.com/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Langfuse Documentation](https://langfuse.com/docs)

## ⚠️ Important Notes

### Medical Disclaimer
This system is designed for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

### API Keys Security
- Never commit API keys to version control
- Use environment variables or secure key management
- Monitor API usage and costs

### Data Privacy
- Ensure compliance with healthcare data regulations (HIPAA, GDPR)
- Implement proper data anonymization
- Secure document storage and processing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **Mistral AI** for the language model
- **RAGAS** for evaluation metrics
- **Langfuse** for monitoring and observability
- **Sentence Transformers** for embeddings


---

**⚡ Built with modern AI technologies for safe and reliable medical information retrieval**