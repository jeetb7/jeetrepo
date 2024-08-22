import logging
import torch  # type: ignore
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings

# Logging and Error Handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_error_and_continue(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper

# Context setup
context = """
Jeet Bera is from Kharagpur. He completed his schooling at Hijli High School and graduated from RCCIIT Kolkata with a degree in Electrical Engineering. Currently, he is working at Xfusion Technologies as a Developer.
"""

# Setup Retrieval Component
@log_error_and_continue
def setup_retrieval_component(text_data, chunk_size=500, chunk_overlap=0, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_text(text_data)
    
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    index = FAISS.from_texts(text_chunks, embeddings)
    return index

# Setup Generative Component
@log_error_and_continue
def setup_generative_component(model_name='gpt2-medium'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Determine if a GPU is available and set the device accordingly
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return generator

@log_error_and_continue
def generate_answer(generator, context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = generator(prompt, max_new_tokens=50)
    return response[0]['generated_text']

# Question-Answering Pipeline
@log_error_and_continue
def qa_pipeline(question, retrieval_index, generator):
    # Retrieve relevant context
    documents = retrieval_index.similarity_search(question)
    if not documents:
        logging.error(f"No relevant documents found for question: {question}")
        return None
    
    context = documents[0].page_content
    
    # Generate answer
    answer = generate_answer(generator, context, question)
    return answer

if __name__ == "__main__":
    # Configuration
    config = {
        "context": context,
        "model_name": "distilgpt2",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 500,
        "chunk_overlap": 0
    }

    # Setup components
    retrieval_index = setup_retrieval_component(config['context'], config['chunk_size'], config['chunk_overlap'], config['embedding_model'])
    generator = setup_generative_component(config['model_name'])

    if retrieval_index is not None and generator is not None:
        # Example questions
        question = "Where did Jeet Bera complete his schooling?"
        answer = qa_pipeline(question, retrieval_index, generator)
        if answer:
            print(answer)
