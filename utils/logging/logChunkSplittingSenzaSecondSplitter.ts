
import { ExtendDocument } from "../../config/RAGJSON.js"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";




//=================================================  Logging e diagnostica  ========================================
/**It's the coordinator that:

Calls createSchemaAwareChunk for intelligent work
Calls RecursiveCharacterTextSplitter if additional splitting is needed
Should log and return results (but currently doesn't)

It's like a manager who delegates specific work to his specialists.

 */


/**    
         WHAT IT DOES: Chunk creation (preprocessing)

Purpose: Prepare documents for indexing!

When: During system setup, before use

Input: Original documents + chunking configuration

Output: Chunks optimized for the vector store


Avoid silent chunking errors that ruin retrieval

Optimize parameters based on real data instead of guesswork

Quick troubleshooting when RAG isn't performing well

Quality control ensures critical metadata isn't lost*/



//Provare o vedere RecursiveCharacterTextSplitter per splittare in chuks i documenti, è una funz di LangChain

//Versione senza SecondSplitter!


export interface ChunkSplittingReturn {
    documents: ExtendDocument[];
    deviceFamilies: Map<string, any>;
}
export async function logChunkSplitting(
    documents: any,
    chunkSize: number,
    chunkOverlap: number,
    recursiveDepth: number
): Promise<ChunkSplittingReturn[]> {

    if (!documents || !Array.isArray(documents)) {
        console.log("documents non è un array, ma è ", typeof documents);
    }
    console.log(`\n=== CHUNK PROCESSING STARTED ===`);
    console.log(`Processing ${documents.length} pre-chunked documents`);
    console.log(`Max chunk size: ${chunkSize}, overlap: ${chunkOverlap}`);

    const MAX_RECURSIVE_DEPTH = 5; //lIMITA LA RICORSIONE

    /*
        //  DEBUG DUPLICAZIONE INPUT
        console.log("🔍 ANALISI INPUT logChunkSplitting:");
        const inputUUIDs = documents.map((d: any) => d.metadata.uuid);
        const uniqueUUIDs = [...new Set(inputUUIDs)];
        console.log(`   Documenti in input: ${documents.length}`);
        console.log(`   UUID unici: ${uniqueUUIDs.length}`);
    */

    // Trova duplicati
    const duplicateMap = new Map();
    documents.forEach((doc: any, idx: any) => {
        const key = `${doc.metadata.uuid}-${doc.metadata.chunkType}`;
        if (!duplicateMap.has(key)) duplicateMap.set(key, []);
        duplicateMap.get(key).push({ index: idx, name: doc.metadata.name });
    });

    duplicateMap.forEach((entries, key) => {
        if (entries.length > 1) {
            console.log(` DUPLICATI trovati per ${key}:`);
            entries.forEach((entry: any) => {
                console.log(`   - Index ${entry.index}: ${entry.name}`);
            });
        }
    });


    const results: { documents: ExtendDocument[]; deviceFamilies: Map<string, any>; }[] | PromiseLike<{ documents: ExtendDocument[]; deviceFamilies: Map<string, any>; }[]> = [];

    const globalDeviceFamilies = new Map<string, any>();

    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
        keepSeparator: true
    });

    // Statistics tracking
    let totalOriginalChunks = 0;
    let totalFinalChunks = 0;
    let chunksRequiringSplitting = 0;
    let chunksRequiringSecondSplit = 0;
    const chunkTypeStats: Record<string, number> = {};


    for (const [index, doc] of documents.entries()) {
        console.log(`\n--- Processing chunk ${index + 1}/${documents.length} ---`);
        console.log(`Chunk type: ${doc.metadata.chunkType}`);
        console.log(`Source: ${doc.metadata.source || 'unknown'}`);
        console.log(`Size: ${doc.pageContent.length} characters`);
        console.log(`Location metadata: floorLocation=${doc.metadata.floorLocation}, isFirstFloor=${doc.metadata.isFirstFloor}, isSecondFloor=${doc.metadata.isSecondFloor}`);


        // Update statistics
        totalOriginalChunks++;
        const chunkType = doc.metadata.chunkType || 'unknown';
        chunkTypeStats[chunkType] = (chunkTypeStats[chunkType] || 0) + 1;

        // Each document is already a chunk - validate and process
        let finalChunks = [doc];


        console.log("Curiosità", finalChunks.length);
        //console.log("Curiosità", JSON.stringify(finalChunks));


        //  ==============================================================================================================
        //                                     Check if this chunk needs secondary splitting
        //  ==============================================================================================================
        if (doc.pageContent.length > chunkSize) {
            console.log(` Chunk oversized (${doc.pageContent.length} > ${chunkSize}), applying secondary splitting...`);
            chunksRequiringSplitting++;

            // Prima suddivisione standard
            const subChunks = await splitter.splitDocuments([doc]);

            // Ricorsione: verifica se i sub-chunks sono ancora troppo grandi
            const recursiveChunk: ExtendDocument[] = [];


            for (const subChunk of subChunks) {
                if (subChunk.pageContent.length > chunkSize) {
                    if (recursiveDepth >= MAX_RECURSIVE_DEPTH) {
                        console.warn("MAx Recursion depth reached, skipping further splitting");
                        finalChunks = [doc];

                    } else {


                        const deeperSplit = await logChunkSplitting([{
                            ...subChunk,
                            metadata: {
                                ...subChunk.metadata,
                                isSubChunk: true,
                                parentChunkId: doc.metadata.chunkId,




                                source: "",
                                loc: "",
                                type: "installation-config",
                                isValid: false,
                                timestamp: "",
                                chunkType: "summary",
                                deviceType: ""
                            }
                        }], chunkSize, chunkOverlap, recursiveDepth + 1);
                        //Recupera i documenti dal risultato
                        for (const result of deeperSplit) {
                            recursiveChunk.push(...result.documents);
                        }
                    }

                } else {
                    recursiveChunk.push(subChunk as ExtendDocument);
                }

            }

            finalChunks = recursiveChunk;
            //Nota: È importante che la ricorsione non si applichi all’intero set di documents,
            //  ma solo a singoli chunk che risultano ancora troppo grandi, altrimenti rischi duplicazioni o cicli infiniti.

        }


        finalChunks = finalChunks.map(chunk => {
            const criticalMetadata = {
                floorLocation: doc.metadata.floorLocation,
                isFirstFloor: doc.metadata.isFirstFloor,
                isSecondFloor: doc.metadata.isSecondFloor,
                partitionNames: doc.metadata.partitionNames,
                areaName: doc.metadata.areaName,
                areaUuid: doc.metadata.areaUuid,
                visualizationType: doc.metadata.visualizationType,
                category: doc.metadata.category,
                chunkType: doc.metadata.chunkType
            };

            return {
                ...chunk,
                metadata: {
                    ...doc.metadata, // Metadati originali come base
                    ...chunk.metadata, // Nuovi metadati
                    ...criticalMetadata // Assicura i metadati critici
                }
            };

        });

        totalFinalChunks += finalChunks.length;

    }

    // Final statistics
    console.log(`\n=== CHUNK PROCESSING COMPLETED ===`);
    console.log(` Processing Statistics:`);
    console.log(`   • Original documents: ${documents.length}`);
    console.log(`   • Total original chunks: ${totalOriginalChunks}`);
    console.log(`   • Total final chunks: ${totalFinalChunks}`);
    console.log(`   • Chunks requiring splitting: ${chunksRequiringSplitting}`);
    console.log(`   • Chunk types distribution:`);
    console.log(" Struttura chunks creati,", JSON.stringify(totalFinalChunks));

    Object.entries(chunkTypeStats).forEach(([type, count]) => {
        console.log(`     - ${type}: ${count}`);
    });


    return results;
}
