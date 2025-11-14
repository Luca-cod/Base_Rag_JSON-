# RAG-LangChain-NaturalLanguageResponseFormat
It's a RAG-Langchain base version, with return a JSON response format.

This is the base implementation of the RAG — the fundamental structure and minimal setup needed to use an LLM via RAG with the LangChain framework.

In this version, I have refactored the chunk-splitting functions to avoid using the hierarchical functions for splitting chunks larger than the maximum size accepted by the embeddings model. I now call the function recursively, using the text splitter from LangChain.


This version serves as a simple foundation for a RAG + Langchain system, into which function modules can be added to improve system performance.
We can see it as a starting point from which to expand the code with various features, as seen in the more extensive versions.