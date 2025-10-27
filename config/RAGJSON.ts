import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Ollama } from "@langchain/ollama";
import { OllamaEmbeddings } from "@langchain/ollama";
import { loadDocumentsJSON } from "../core/retrieval/loaders/loadDocumentJSON3 copy.js";
import { Chroma } from "@langchain/community/vectorstores/chroma"; //Alternativa a FAISS
import { RunnableMap, RunnablePassthrough } from "@langchain/core/runnables";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import path from "path";
import { Document as LangChainDocument } from "langchain/document";
import { EndpointMetadata } from "../core/retrieval/loaders/loadDocumentJSON3 copy.js";

export const config = {
  documentPath: "/home/luca/RagBaseNLP/src/data/", //Cartella con i documenti
  faissIndexPath: "./faiss_index", // Path per salvare l'indice FAISS
  outputPath: "/home/luca/RagBaseNLP/src/response/", // Cartella per salvare le risposte JSON
  modelName: "llama3.2:1b", // Nome modello Ollama
  chunkSize: 1000, // Dimensione chunk per lo splitting
  chunkOverlap: 250, // Overlap tra chunk
  retrievalConfig: {
    k: 15
  },
  jsonSplitting: {
    splitKeys: ['endpoints', 'actions', 'rules'], // Chiavi da splittare
    preserveKeys: ['manifest'] // Chiavi da mantenere intere
  }
};
//llama3.2:3b
export const llm = new Ollama({
  baseUrl: "http://localhost:11434",
  model: config.modelName,
  temperature: 0.01, //Valore più alto = risposte più creative, valore più baso = risposte più concrete
  format: "json",  //per formato JSon della risposta, se qua formato JSON nel prompt utilizzo JSONOutputParser
  numCtx: 4097, //Aumenta il contesto se possibile
  topP: 0.95 //Per maggiore coerenza
});

export const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",  // URL di Ollama
  model: "nomic-embed-text",
  maxRetries: 2,
  maxConcurrency: 3,//Riduci per concorrenza di debug
});


export interface ExtendDocument extends LangChainDocument {
  metadata: EndpointMetadata,
  readableText?: string; //opzionale per testo leggibile
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



//const model = new SentenceTransformer('paraphrase-MiniLM-L6-v2');
export const targetFile = 'installation-config.json'; //File specifico da processare
export const directoryPath = "/home/luca/ragts2GeneralQuery New/src/data";
export const filePath = path.join(directoryPath, targetFile);




//export const user_query = "Give me a list of the uuid from the 'sensor' devices in the configuration. Indicate the name and category.";
//export const user_query = "Dimmi che dispositivi ci sono nel file";
//export const user_query = "Che sensore mi può dare informazioni di temperatura?";
//export const user_query = "Endpoint per il controllo luci";
/***"Come si chiama il termostato"?
"Che sensore mi può dare informazioni di temperatura"? 
"Esistono dei sensori per controllare luci"? */
//export const user_query = "Tell me something about the controller and the firmaware version"; //---> funziona
//export const user_query = "What is the name of thermostat?"
//export const user_query = "Dimmi tutti i parametri del termostato"
//export const user_query = "Give me all parameters of BOX-IO";
//export const user_query = "Show me all the sensors connected to the first floor.";
//export const user_query = "Show me sensors";
//export const user_query = "Show me devices";
//export const user_query = "Accendi le luci";
//export const user_query = "What is the default thermostat setpoint?"; // il "default" rischia di far si che il modello non se la senta di "inventare" perchè non abbiamo un parametro "default" il valore da attribuire
//export const user_query = "What is the value of the thermostat setpoint?";
//export const user_query = "Show me all devices located on the second floor";
export const user_query = "Show me the UUIDs of actuator, thermostat and controller";
//export const user_query = "Dimmi qual'è l' UUID del controller luci soggiorno";

/**A. Test con Query Tipiche:
- "Mostra i sensori di temperatura"
- "UUID del controller luci soggiorno"  
- "Parametri configurabili termostato"
- "Dispositivi zona cucina" */


/**                           QUERY PER AUTOMAZIONE, DA TESTARE 
 *     "Create an automation for the thermostat"

    "When the temperature exceeds 25°, turn on the air conditioner"

    "Schedule the lights to turn on at 6:00 PM"

    "If there is motion, turn on the lights" 
    
    */



async function main() {
  try {
    await runRgaSytsem(user_query);
  } catch (error) {
    console.log("Errore nell'esecuzione del RAG...");
  }
}

async function runRgaSytsem(query: string) {

  try {

    const Document = loadDocumentsJSON();

    //Create a Vector Store and load FAISS index
    let vectoreStore: FaissStore;
    vectoreStore = await FaissStore.load(config.faissIndexPath, embeddings);

    let k = config.retrievalConfig.k;



    //  retrievalDocs = await vectoreStore.similaritySearch(user_query, config.retrievalConfig.k);

    /**Versione con asRetrieval
     **/
    const retriever = vectoreStore.asRetriever({
      k,
      searchType: "similarity", //Non supportrato "mmr"
      verbose: true
    });

    const retrievalDocs = await retriever.invoke(user_query);
    console.log("Quanti documenti ha recuperato l'asRetriever? Ecco qua:", retrievalDocs.length);

    const filteringDocs = filterByContentRelevance(retrievalDocs, user_query, 0.3);

    console.log("Quanti documenti otteniamo dopo il filtro per rilevanza?", filteringDocs.length);

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
      query: new RunnablePassthrough()//RunnablePassthrough.isRunnable((input: { query: string }) => input.query),
    })
      .pipe(prompt)
      .pipe(llm);

    const response = await chain.invoke({ query: "Tell me something about the controller and the firmaware version" });

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

    // Normalizza il score per lunghezza query
    const relevanceScore = score / queryTerms.length;
    return relevanceScore >= threshold;
  });
}

main();