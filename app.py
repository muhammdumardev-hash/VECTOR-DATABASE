import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing Chroma database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="university_faqs"
)

# User questions with different wording
questions = [
    "How can I sign up for my subjects?",
    "Where can I get help if my computer is not working?",
    "How can I see my academic marks?"
]

print("=" * 65)
print("DAY 9 - FINAL SEMANTIC SEARCH EXPERIMENT")
print("=" * 65)

for question in questions:

    # Convert question into embedding
    question_embedding = model.encode(question).tolist()

    # Search Chroma database
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=1
    )

    print("\nUser Question:")
    print(question)

    print("\nMost Relevant Information:")
    print(results["documents"][0][0])

    print("\nMetadata:")
    print(results["metadatas"][0][0])

    print("-" * 65)

print("\nSemantic search experiment completed successfully.")