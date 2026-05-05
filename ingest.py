import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

def extract_pdf_data(file_path):
    print(f"Starting LlamaParse for: {file_path}")
    print("This might take some time depending on the PDF size...")
    
    # Initialize the parser
    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        raise ValueError("Missing LLAMA_CLOUD_API_KEY in environment variables")
    
    parser = LlamaParse(
        api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True
    )
    
    parsed_docs = parser.load_data(file_path)
    
    output_filename = "parsed_output.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        for doc in parsed_docs:
            f.write(doc.text)
            f.write("\n\n---PAGE BREAK---\n\n")
            
    print(f"Success! Check {output_filename} to see your cleanly parsed data.")
    return parsed_docs

if __name__ == "__main__":
    pdf_path = "./data/sample_10k.pdf" 
    
    if os.path.exists(pdf_path):
        documents = extract_pdf_data(pdf_path)
    else:
        print(f"Error: Could not find {pdf_path}")