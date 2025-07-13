# RAG system example
This project is a simple example of the process involved in creating a Retrieval Augmented Generation system.  This project specifically uses 3 scientific research papers that are on the topic of VO2 max.  I'm only using three papers on a narrow topic to keep the project small.  

## PDF_to_text.ipynb
One key challenge is to obtain a sufficient volume of research papers in text format.  These papers are often in PDF format so we'll handle extracting text from PDFs.  The quality of text extraction from PDFs can vary. We'll do some cleaning and preprocessing so that the RAG system can focus on the main content.

This jupyter notebook extracts the text from 3 pdfs and saves as 3 text files in the 'extracted_texts' folder, then does some cleaning and preprocessing to the 3 text files and saves them in the 'preprocessed_texts' folder.  

## main.ipynb 
This file handles the steps involved in RAG systems: Chunking, Embedding, and storing the embeddings in a vector database to allow for efficient similarity search.  

Main is a jupyter notebook that works through the steps involved in a RAG system using the text from the preprocessed files.  

Generating a response from an LLM requires an API key from the LLM provider and often has pretty stringent limits on use at the free tier.  I plan to explore this further as there are some methods for moderating use according to those limits that may allow it to work.
