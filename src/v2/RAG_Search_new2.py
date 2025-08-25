import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
from langchain import hub
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.agents import initialize_agent, AgentType, create_tool_calling_agent, AgentExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool
import asyncio
from asyncio import timeout
import logging
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

conversation_history = []

def create_vector_store_from_pdfs(folder_path: str, vector_store_path: str, index_name: str = 'cpm-index',
                                  model_name: str = 'nomic-embed-text', chunk_size: int = 500, chunk_overlap: int = 0):
    """
    Load PDFs from a directory, split them into chunks, and store them as vectors.

    :param folder_path: Path to the folder containing PDFs.
    :param vector_store_path: Path to save the vector store.
    :param index_name: Name of the index.
    :param model_name: Name of the embedding model.
    :param chunk_size: Size of the text chunks.
    :param chunk_overlap: Overlap between text chunks.
    """
    try:
        # Validate inputs
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"PDF folder not found: {folder_path}")
        
        # Initialize embeddings model
        underlying_embeddings = OllamaEmbeddings(model=model_name)
        # Load documents from the specified folder
        loader = DirectoryLoader(folder_path, show_progress=True, loader_cls=PyPDFLoader)
        data = loader.load()
        
        if not data:
            raise ValueError("No documents found in the specified folder")
     
        # Log the number of documents loaded
        print(f"Loaded {len(data)} documents from {folder_path}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(data)
        print(f"Split documents into {len(documents)} chunks")
        
        # Create vector store from documents
        vectorstore = FAISS.from_documents(documents, underlying_embeddings)
#        print(f"Created vector store with {len(vectorstore)} vectors")

        # Save the vector store locally
        vectorstore.save_local(folder_path=vector_store_path, index_name=index_name)
    except Exception as e:
        logging.error(f"Vector store creation failed: {str(e)}")
        raise

@lru_cache(maxsize=32)
def load_vector_store(folder_path, embeddings, index_name='cpm-index'):
    """
    Load a vector store from local storage.

    :param folder_path: Path to the folder containing the vector store.
    :param embeddings: Embedding model used for the vector store.
    :param index_name: Name of the index.
    :return: Loaded vector store.
    """
    print(f"Loading vector store from {folder_path} with index name {index_name}")
    return FAISS.load_local(folder_path=folder_path, embeddings=embeddings, index_name=index_name,
                            allow_dangerous_deserialization=True)


def create_retriever_tool_from_vector(vector, tool_name="EIS_search"):
    """
    Create a retriever tool from a vector store.

    :param vector: The vector store.
    :param tool_name: Name of the tool.
    :return: Configured retriever tool.
    """
    retriever = vector.as_retriever()
    return create_retriever_tool(
        retriever,
        tool_name,
        "Search for information about EIS. For any questions about EIS, you must use this tool!",
    )


def create_agent_executor(model, tools, prompt):
    """
    Create an agent executor.

    :param model: Chat model.
    :param tools: List of tools to bind to the model.
    :param prompt: Prompt template to be used.
    :return: Configured agent executor.
    """
    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


def RAG_search(input_text, history):
    """Main function to set up and execute the agent."""
    try:
        if not os.path.exists('vector_path/FAISS.db'):
            raise FileNotFoundError("Vector store not found. Please create vector store first.")
            
        underlying_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector = load_vector_store(
            folder_path='vector_path/FAISS.db',
            embeddings=underlying_embeddings,
            index_name='cpm-index'
        )
        local_search = create_retriever_tool_from_vector(vector)
        tools = [local_search]

        model = ChatOllama(model="llama3.1", temperature=0.7)

        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent_executor = create_agent_executor(model, tools, prompt)

        # Combine history and current input
        combined_input = "\n".join(history + [input_text])

        # Add timeout handling
        with timeout(30):  # 30 seconds timeout
            output = agent_executor.invoke({"input": combined_input})
            
        return output
        
    except TimeoutError:
        return "Search operation timed out. Please try again."
    except Exception as e:
        logging.error(f"RAG search failed: {str(e)}")
        raise e


class Application(tk.Tk):
    def __init__(self, circuit_string=""):
        super().__init__()
        self.title("Vector Store and RAG Search Tool")

        # Set the circuit_string
        self.circuit_string = circuit_string

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # LangSmith API
        self.api_label = tk.Label(self, text="LangSmith API:")
        self.api_label.grid(row=0, column=0, sticky='e')
        self.api_entry = tk.Entry(self, width=50)
        self.api_entry.grid(row=0, column=1, columnspan=2)
        self.api_entry.insert(0, "...")

        # Folder path for PDFs
        self.pdf_folder_label = tk.Label(self, text="PDF Folder Path:")
        self.pdf_folder_label.grid(row=1, column=0, sticky='e')
        self.pdf_folder_entry = tk.Entry(self, width=50)
        self.pdf_folder_entry.grid(row=1, column=1)
        self.pdf_folder_button = tk.Button(self, text="Browse", command=self.browse_pdf_folder)
        self.pdf_folder_button.grid(row=1, column=2)

        # Vector store path
        self.vector_store_label = tk.Label(self, text="Vector Store Path:")
        self.vector_store_label.grid(row=2, column=0, sticky='e')
        self.vector_store_entry = tk.Entry(self, width=50)
        self.vector_store_entry.grid(row=2, column=1)
        self.vector_store_button = tk.Button(self, text="Browse", command=self.browse_vector_store)
        self.vector_store_button.grid(row=2, column=2)

        # Load vector store path
        self.load_vector_store_label = tk.Label(self, text="Load Vector Store Path:")
        self.load_vector_store_label.grid(row=3, column=0, sticky='e')
        self.load_vector_store_entry = tk.Entry(self, width=50)
        self.load_vector_store_entry.grid(row=3, column=1)
        self.load_vector_store_entry.insert(0, "/vector_path/cpm-index.faiss") 
        self.load_vector_store_button = tk.Button(self, text="Browse", command=self.browse_load_vector_store)
        self.load_vector_store_button.grid(row=3, column=2)

        # Input text
        self.input_label = tk.Label(self, text="Input:")
        self.input_label.grid(row=4, column=0, sticky='ne')
        self.input_text = scrolledtext.ScrolledText(self, width=50, height=10)
        self.input_text.grid(row=4, column=1, columnspan=2)
        self.input_text.insert(tk.END, self.circuit_string)

        # Output text
        self.output_label = tk.Label(self, text="Output:")
        self.output_label.grid(row=5, column=0, sticky='ne')
        self.output_text = scrolledtext.ScrolledText(self, width=50, height=10)
        self.output_text.grid(row=5, column=1, columnspan=2)

        # Buttons
        self.create_vector_store_button = tk.Button(self, text="Create Vector Store", command=self.create_vector_store)
        self.create_vector_store_button.grid(row=6, column=1, sticky='e')

        self.run_search_button = tk.Button(self, text="Run RAG Search", command=self.run_search)
        self.run_search_button.grid(row=6, column=2, sticky='w')

        # self.history_button = tk.Button(self, text="Show Conversation History", command=self.show_conversation_history)
        # self.history_button.grid(row=7, column=1, sticky='e')

    def browse_pdf_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.pdf_folder_entry.delete(0, tk.END)
            self.pdf_folder_entry.insert(0, folder_path)

    def browse_vector_store(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.vector_store_entry.delete(0, tk.END)
            self.vector_store_entry.insert(0, folder_path)

    def browse_load_vector_store(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.load_vector_store_entry.delete(0, tk.END)
            self.load_vector_store_entry.insert(0, folder_path)

    def create_vector_store(self):
        api_key = self.api_entry.get().strip()
        os.environ["LANGCHAIN_API_KEY"] = api_key

        folder_path = self.pdf_folder_entry.get()
        vector_store_path = self.vector_store_entry.get()
        if not folder_path or not vector_store_path:
            messagebox.showerror("Error", "Please specify both PDF folder path and vector store path.")
            return
        create_vector_store_from_pdfs(folder_path, vector_store_path)
        messagebox.showinfo("Success", "Vector store created successfully.")

    def run_search(self):
        default_api_key = ""
        api_key = self.api_entry.get().strip() or default_api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key

        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showerror("Error", "Please enter input text for the search.")
            return

        output_data = RAG_search(input_text, conversation_history)

        self.output_text.delete("1.0", tk.END)
        latest_conversation = f"User: {input_text}\nLLM: {output_data}"
        self.output_text.insert(tk.END, latest_conversation)

        conversation_history.append(f"User: {input_text}")
        conversation_history.append(f"LLM: {output_data}")

        self.input_text.delete("1.0", tk.END)

    # def show_conversation_history(self):
    #     history_window = tk.Toplevel(self)
    #     history_window.title("Conversation History")

    #     history_text = scrolledtext.ScrolledText(history_window, wrap=tk.WORD, width=80, height=40)
    #     history_text.pack(pady=5)

    #     full_conversation = "\n".join(conversation_history)
    #     history_text.insert(tk.END, full_conversation)
    #     history_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    import sys
    circuit_string = sys.argv[1] if len(sys.argv) > 1 else ""
    app = Application(circuit_string=circuit_string)
    app.mainloop()
