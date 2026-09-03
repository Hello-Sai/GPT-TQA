


from loaders import DocumentLoader 
from chunkers import DocumentChunker
from embeddings import HuggingFaceEmbeddingsModel
from vector_store import VectorStore
from llms import Retriever, ResponseGenerator 

class RAGPipeline:
    # Entry Point for RAG Pipeline Class Embedding all the Key Components ---
    def __init__(self, doc_path: str):
        # Initialization done with all components
        self.loader = DocumentLoader(doc_path) 
        self.chunker = DocumentChunker()
        self.embeddings = HuggingFaceEmbeddingsModel()
        self.vectorstore = None
        self.retriever = None
        self.generator = None
        self.qa_chain = None

    def build(self):
        """Setup Incurs : Loads data, chunks it, creates embeddings, and builds the vector store/chain."""
        print("\n--- Starting RAG Pipeline Build ---")
        
        # 1. Load Documents
        documents = self.loader.load_content()
        if not documents:
            print("No content loaded. Exiting build.")
            return
            
        # 2. Chunk Documents
        chunks = self.chunker.create_chunks(documents)
        
        # 3. Create Vector Store 
        vector_store_manager = VectorStore(self.embeddings.get_embeddings())
        self.vectorstore = vector_store_manager.create_store(chunks)
        
        # 4. Create Retriever and QA Chain
        retriever_component = Retriever(self.vectorstore)
        self.retriever = retriever_component.retriever
        
        self.generator = ResponseGenerator()
        self.qa_chain = self.generator.create_qa_chain(self.retriever)
        
        print("--- RAG Pipeline Ready ---")
            
    def query(self, question: str) -> str:
        if not self.qa_chain:
            return "Error: RAG Pipeline is not built. Run pipeline.build() first."
            
        # The RetrievalQA chain accepts a dictionary with the key 'query'
        prompt_payload = {"query": question}
        
        # Execute the QA chain
        response = self.qa_chain(prompt_payload)
        
        # Format and return the response
        result = response.get('result', "No answer found.")

        output = f"\n🤖 Answer:\n{result}\n"
        
        
            
        return output

def main():
    # File path for the local speech document
    doc_path = "speech.txt"
    
    # 1. Initialize and Build the Pipeline
    pipeline = RAGPipeline(doc_path)
    pipeline.build()
    
    if not pipeline.qa_chain:
        print("Pipeline failed to build successfully. Please check dependencies.")
        return

    # 2. Command-Line Q&A Loop
    print("\n--- Command-Line Q&A Interface ---")
    print("Ask a question about the document (type 'exit' or 'quit' to stop).")
    
    while True:
        try:
            query_text = input("You: ").strip()
            
            if query_text.lower() in ["exit", "quit"]:
                print("Exiting Q&A. Goodbye!")
                break
            
            if not query_text:
                continue

            response = pipeline.query(query_text)
            print(response)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()
