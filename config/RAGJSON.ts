import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Ollama } from "@langchain/ollama";
import { OllamaEmbeddings } from "@langchain/ollama";
import { loadDocumentsJSON, EndpointMetadata } from "../core/retrieval/loaders/loadDocumentJSON.js";
import { RunnableMap, RunnablePassthrough } from "@langchain/core/runnables";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import path from "path";
import { Document as LangChainDocument } from "langchain/document";
import { user_query } from "../config/Query.js";
import { promises as fs } from "fs";
import { logChunkSplitting } from "src/utils/logging/logChunkSplittingSenzaSecondSplitter.js";


export const config = {
  documentPath: "/home/luca/RagBaseJSON/src/data/", // Folder containing the documents
  faissIndexPath: "./faiss_index", // Path to save the FAISS index
  outputPath: "/home/luca/RagBaseJSON/src/response/", // Folder to save JSON responses
  modelName: "llama3.2:1b", // Ollama model name
  chunkSize: 1000, // Chunk size for text splitting
  chunkOverlap: 250, // Overlap between chunks
  retrievalConfig: {
    k: 15
  },
  jsonSplitting: {
    splitKeys: ['endpoints', 'actions', 'rules'], // Keys to split
    preserveKeys: ['manifest'] // Keys to keep intact
  }
};

// llama3.2:3b
export const llm = new Ollama({
  baseUrl: "http://localhost:11434",
  model: config.modelName,
  temperature: 0.01, // Higher = more creative responses, lower = more factual
  format: "json",  // Forces the model to respond in JSON format
  numCtx: 4097, // Increase context size if possible
  topP: 0.95 // For greater coherence
});

export const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",  // Ollama API URL
  model: "nomic-embed-text",
  maxRetries: 2,
  maxConcurrency: 3, // Reduce if debugging concurrency issues
});


export interface ExtendDocument extends LangChainDocument {
  metadata: EndpointMetadata,
  readableText?: string; // Optional readable text field
}



const prompt = PromptTemplate.fromTemplate(`
    You are a home automation data analyst. Your job is to extract and present device information from the provided context in valid JSON format.
    
    === CRITICAL RULES ===
    
    1. **ONLY USE DATA FROM CONTEXT** - Never add information from general knowledge
    2. **VALID JSON ONLY** - No markdown, no explanations, no additional text
    3. **EXACT VALUES** - Copy values literally as they appear in the context
    4. **MISSING DATA** - If information is not in context, omit it from JSON (don't use null/empty strings)
    
    === CONTEXT DATA ===
    
    {context}
    
    === USER QUERY ===
    
    {query}
    
    === RESPONSE FORMAT ===
    
    Return a JSON object with this exact structure:
    
    {{
      "devices": [
        {{
          "name": "Device Name",
          "uuid": "device-uuid",
          "category": "Category Name",
          "location": "Floor location",
          "parameters": [
            {{
              "name": "parameter_name",
              "value": <actual_value>,
              "unit": "unit_if_present",
              "dataType": "type_from_context"
            }}
          ]
        }}
      ]
    }}
    
    === EXAMPLES ===
    
    **Query:** "Show devices on second floor"
    **Context contains:** Line 1 lights (second floor, 5 params), Energy meter (second floor, 68 params)
    
    **Correct response:**
    {{
      "devices": [
        {{
          "name": "Line 1 lights",
          "uuid": "eaf97aeb-1a34-40e5-8197-7d1b7fdb4b49",
          "category": "Controller",
          "location": "Second floor",
          "parameters": [
            {{"name": "accensione", "value": false, "dataType": "boolean"}},
            {{"name": "livello", "value": 100, "dataType": "number"}},
            {{"name": "identifica", "value": false, "dataType": "boolean"}}
          ]
        }},
        {{
          "name": "Energy meter",
          "uuid": "2beaf79c-3c02-4282-9bf3-4af1cda1f676",
          "category": "Actuator",
          "location": "Second floor",
          "parameters": [
            {{"name": "phase_1_power", "value": 0, "unit": "watt", "dataType": "decimal"}},
            {{"name": "phase_2_power", "value": 0, "unit": "watt", "dataType": "decimal"}},
            {{"name": "total_system_power", "value": 0, "unit": "watt", "dataType": "decimal"}}
          ]
        }}
      ]
    }}
    
    === PARAMETER HANDLING ===
    
    **If device has ≤15 parameters:** Include all parameters  
    **If device has >15 parameters:** Include first 10 + last 5 most relevant
    
    For devices with many parameters (like Energy meter with 68):
    - Prioritize: control parameters, measurements with non-zero values, named setpoints
    - Group similar parameters (e.g., all phases together)
    - Always include: connectivity status, primary measurements
    
    **Device NOT in context:**
    {{
      "devices": [],
      "note": "No devices matching query found in provided context"
    }}
    
    === VERIFICATION CHECKLIST ===
    
    Before responding, verify:
    - ✓ Every device name is literally from context
    - ✓ Every UUID is exactly as shown
    - ✓ Every parameter name matches context spelling
    - ✓ Every value is from context (not assumed/calculated)
    - ✓ JSON is valid (test with JSON.parse)
    - ✓ No devices from general knowledge added
    
    === OUTPUT ===
    
    Return ONLY the JSON object. No prefix, no markdown code blocks, no explanations.
  `);


// const model = new SentenceTransformer('paraphrase-MiniLM-L6-v2');
export const targetFile = 'installation-config.json'; // Specific file to process
export const directoryPath = "/home/luca/RagBaseJSON/src/data";
export const filePath = path.join(directoryPath, targetFile);



async function main() {
  try {
    let response = await runRgaSystem(user_query);

    await saveResponse(user_query, response);

  } catch (error) {
    console.log("Error while running the RAG system...");
  }
}

async function runRgaSystem(query: string) {

  try {
    const { Documents } = await loadDocumentsJSON(); //Destruct 
    const depth = 5;
    const DocsSplit = logChunkSplitting(Documents, config.chunkSize, config.chunkOverlap, depth);

    console.log("Dimension chunks", ((await DocsSplit).length))


    // Create a Vector Store and load FAISS index
    let vectorStore: FaissStore;
    vectorStore = await FaissStore.load(config.faissIndexPath, embeddings);

    let k = config.retrievalConfig.k;

    /** Version using asRetriever **/
    const retriever = vectorStore.asRetriever({
      k,
      searchType: "similarity", // "mmr" not supported
      verbose: true
    });

    const retrievalDocs = await retriever.invoke(user_query);
    console.log("How many documents did asRetriever recover? Here:", retrievalDocs.length);

    const filteringDocs = filterByContentRelevance(retrievalDocs, user_query, 0.3);

    console.log("How many documents remain after relevance filtering?", filteringDocs.length);

    const context = filteringDocs.map((doc: any) => doc.pageContent).join("\n\n");

    /*const chain = {
      {
      "context": retrievalDocs.map(doc => doc.pageContent).join("\n\n"),
        "query": RunnablePassthrough.from(async (input: { query: string }) => input.query),
        
      }
      | prompt={ `You are an expert system for home automation configuration. Using the provided context, answer the following query in a concise manner.` }
        | model: llm
      | StringOutputParser
    */

    const chain = RunnableMap.from({
      context: () => context,
      query: new RunnablePassthrough()
    })
      .pipe(prompt)
      .pipe(llm);

    const response = await chain.invoke({ query: "Tell me something about the controller and the firmware version" });

    console.log(response);

    return response;

  } catch (error) {
    console.log("Error in RAG system:", error);
    return { system: "error" };
  }

}


function filterByContentRelevance(docs: string | any, query: string, threshold = 0.3) {
  const queryTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2);

  return docs.filter((doc: any) => {
    const content = doc.pageContent.toLowerCase();
    let score = 0;

    queryTerms.forEach(term => {
      if (content.includes(term)) score += 1;
    });

    // Normalize score based on query length
    const relevanceScore = score / queryTerms.length;
    return relevanceScore >= threshold;
  });
}

async function saveResponse(query: string, response: any) {

  //const responseText = typeof response === 'string' ? response : JSON.stringify(response, null, 2);

  console.log("Starting response save...");

  try {
    // Create folder if it doesn’t exist
    await fs.mkdir(config.outputPath, { recursive: true });
    // Create unique filename
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `response_${timestamp}.json`;
    const fullPath = path.join(config.outputPath, filename);

    // Save file
    await fs.writeFile(
      fullPath,
      response,
      "utf-8"
    );

    console.log("Response saved at:", fullPath);
    console.log("Executed query:", query);

  } catch (error) {
    console.error("Error saving response:", error);
    throw error;
  }
}

main();
