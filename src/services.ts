import { GoogleGenAI, HarmCategory, HarmBlockThreshold, Type, Part } from "@google/genai";
import { v4 as uuidv4 } from 'uuid';
import * as dotenv from 'dotenv';
import pdfParse from 'pdf-parse';
import { Pinecone } from '@pinecone-database/pinecone';
import { getToken } from "./utils/token";
import { ZOOM_API_BASE_URL } from "./constants";
import { ragFunctionDeclaration, weatherFunctionDeclaration, zoomFunctionDeclaration } from "./constants/tools";

// Configure dotenv before any other imports that need env variables
dotenv.config();

// --- ENVIRONMENT VARIABLES ---
// IMPORTANT: Create a .env file in your root directory and add your API key.
// GOOGLE_API_KEY=your_api_key_here
const API_KEY = process.env.GOOGLE_API_KEY as string;

const PROJECT = process.env.GOOGLE_PROJECT as string;
const LOCATION = process.env.GOOGLE_LOCATION as string;
const MODEL = process.env.GOOGLE_MODEL as string;

const PINECONE_API_KEY = process.env.PINECONE_API_KEY as string;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME as string; // You'll need to add this to your .env
const WEATHER_API = process.env.WEATHER_API_KEY as string;

// --- MODEL & AI SETUP ---
const genAI = new GoogleGenAI({ apiKey: API_KEY });
const vertexAI = new GoogleGenAI({
    vertexai: true,
    project: PROJECT,
    location: LOCATION,
});

const generationConfig = {
    temperature: 0.9,
    topK: 1,
    topP: 1,
    maxOutputTokens: 2048,
};

const safetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.OFF },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.OFF },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.OFF },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.OFF },
];

const systemInstruction = `Remember today is ${new Date()}.`;

// --- VECTOR DATABASE ---

let pinecone: Pinecone;

async function initPinecone() {
    if (!pinecone) {
        pinecone = new Pinecone({
            apiKey: PINECONE_API_KEY,
        });
    }
    return pinecone;
}

// --- UTILITY FUNCTIONS ---

/**
 * Splits a long text into smaller chunks of a specified size.
 * @param text The text to split.
 * @param chunkSize The desired size of each chunk.
 * @returns An array of text chunks.
 */
function chunkText(text: string, chunkSize = 1000): string[] {
    const chunks: string[] = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.substring(i, i + chunkSize));
    }
    return chunks;
}


// --- FILE PROCESSING & EMBEDDING ---

/**
 * Handles the upload of a file, chunks it, creates embeddings, and stores them in the vector DB.
 * @param fileBuffer The buffer containing the file data.
 * @param originalname The original name of the file.
 * @returns The ID of the document.
 */
export async function handleFileUpload(fileBuffer: Buffer, originalname: string): Promise<string> {
    console.log(`Processing file: ${originalname}`);
    const documentId = uuidv4();

    // 1. Read and chunk the text
    const text = await pdfParse(fileBuffer);
    console.log(text)
    const chunks = chunkText(text.text);
    console.log(`File split into ${chunks.length} chunks.`);

    // 2. Generate and store embeddings for each chunk in Pinecone
    const pineconeClient = await initPinecone();
    const index = pineconeClient.Index(PINECONE_INDEX_NAME);

    const vectorsToUpsert = [];

    for (const chunk of chunks) {
        const result = await genAI.models.embedContent(
            {
                model: "gemini-embedding-001",
                contents: { parts: [{ text: chunk }] },
                config: {
                    taskType: "RETRIEVAL_DOCUMENT",
                    outputDimensionality: 1024,
                },
            }
        );
        const embedding = result.embeddings[0].values;

        vectorsToUpsert.push({
            id: uuidv4(),
            values: embedding,
            metadata: { text: chunk, documentId: documentId, originalname: originalname },
        });
    }

    // Upsert vectors in batches (Pinecone recommends batches of 100)
    const batchSize = 100;
    for (let i = 0; i < vectorsToUpsert.length; i += batchSize) {
        const batch = vectorsToUpsert.slice(i, i + batchSize);
        await index.upsert(batch);
    }

    console.log(`Successfully created and stored ${vectorsToUpsert.length} vectors in Pinecone for document ${originalname}.`);
    return documentId;
}


// --- CHAT & RETRIEVAL-AUGMENTED GENERATION (RAG) ---

/**
 * Handles a chat query by searching the vector DB for context and generating a response.
 * @param query The user's question.
 * @param history The previous chat history.
 * @returns The generated response from the Gemini model.
 */
export async function handleChat(query: string, history: any[]): Promise<any> {
    console.log(`Handling chat query: "${query}"`);

    // 1. Embed the user's query
    const queryEmbeddingResult = await genAI.models.embedContent(
        {
            model: "gemini-embedding-001",
            contents: { parts: [{ text: query }] },
            config: {
                taskType: "RETRIEVAL_QUERY",
                outputDimensionality: 1024,
            }
        }
    );
    const queryEmbedding = queryEmbeddingResult.embeddings[0].values;

    // 2. Query Pinecone for similar documents
    const pineconeClient = await initPinecone();
    const index = pineconeClient.Index(PINECONE_INDEX_NAME);

    const queryResponse = await index.query({
        vector: queryEmbedding,
        topK: 5, // Get top 5 most similar chunks
        includeMetadata: true,
    });

    const similarDocs = queryResponse.matches?.map(match => match.metadata?.text).filter(Boolean) || [];

    const context = similarDocs.join(`

---

`);
    console.log(`Found ${similarDocs.length} relevant document chunks from Pinecone.`);

    const prompt = `
        Based on the following context, please answer the user's question.
        If the context does not contain the answer, say that you don't have enough information.

        CONTEXT:
        ${context}

        QUESTION:
        ${query}
    `;

    // 3. Prepare the prompt for the chat model
    const chatModel = await genAI.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: {
            ...generationConfig,
            safetySettings: safetySettings
        },
    });

    // 4. Generate the response

    console.log("Generated response successfully.");
    return chatModel.text;
}

/**
 * Handles a chat query by searching the vector DB for context and generating a response.
 * @param query The user's question.
 * @param history The previous chat history.
 * @returns The generated response from the Gemini model.
 */
export async function handleGenerate(query: string, history: any[]): Promise<string> {
    const chatModel = await genAI.models.generateContent({
        model: "gemini-2.5-flash",
        contents: [
            ...history,
            {
                parts: [{
                    text: query
                }],
                role: "user"
            }
        ],
        config: {
            systemInstruction,
            ...generationConfig,
            safetySettings,
            tools: [{
                functionDeclarations: [weatherFunctionDeclaration, zoomFunctionDeclaration, ragFunctionDeclaration]
            }]
        },
    });

    // Check for function calls in the response
    if (chatModel.functionCalls && chatModel.functionCalls.length > 0) {
        let results: Part[] = [];

        for (const functionCall of chatModel.functionCalls) {
            console.log(`Function to call: ${functionCall.name}`);
            console.log(`Arguments: ${JSON.stringify(functionCall.args)}`);
            const funcArgs: any = functionCall.args;

            if (functionCall.name === 'get_current_temperature') {
                const res = await fetch(`http://api.weatherstack.com/current?access_key=${WEATHER_API}&query=${encodeURI(funcArgs.location)}`, {
                    headers: {
                        'Content-Type': 'application/json',
                    },
                }).then(response => response.json());

                results.push({
                    functionResponse: {
                        name: functionCall.name,
                        response: {
                            result: res // The 'response' needs to be a dict/object
                        }
                    }
                })
            } else if (functionCall.name === 'create_zoom_meeting') {
                const token = await getToken();

                const res = await fetch(
                    `${ZOOM_API_BASE_URL}/users/me/meetings`,
                    {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            Authorization: `Bearer ${token.access_token}`,
                        },
                        body: JSON.stringify({
                            topic: funcArgs.topic,
                            settings: {
                                meeting_invitees: funcArgs.meeting_invitees,
                            },
                            start_time: funcArgs.start_time,
                            timezone: "Asia/Manila"
                        }),
                    }
                ).then(response => response.json());

                results.push({
                    functionResponse: {
                        name: functionCall.name,
                        response: {
                            result: res // The 'response' needs to be a dict/object
                        }
                    }
                })
            } else if (functionCall.name === 'rag_intro_llm_prompt_engineering') { // change the function name according to what data source you uploaded.
                const res = await handleChat(query, history);

                results.push({
                    functionResponse: {
                        name: functionCall.name,
                        response: {
                            result: res // The 'response' needs to be a dict/object
                        }
                    }
                })
            }
        }

        const functionRes = await genAI.models.generateContent({
            model: "gemini-2.5-flash",
            contents: [
                ...history,
                {
                    parts: [{
                        text: query
                    }],
                    role: "user"
                },
                {
                    parts: results
                }
            ],
            config: {
                ...generationConfig,
                safetySettings: safetySettings,
            }
        });

        return functionRes.text;
        // In a real app, you would call your actual function here:
        // const result = await getCurrentTemperature(functionCall.args);
    } else {
        console.log("No function call found in the response.");

        return chatModel.text;
    }
}

/**
 * Handles a chat query using a fine-tuned model.
 * @param query The user's question.
 * @param history The previous chat history.
 * @returns The generated response from the fine-tuned model.
 */
export async function handleFineTuned(query: string, history: any[]): Promise<string> {
    try {
        console.log(`Handling fine-tuned query: "${query}"`);

        const chatModel = await vertexAI.models.generateContent({
            model: MODEL,
            contents: [
                ...history,
                {
                    parts: [{
                        text: query
                    }],
                    role: "user"
                }
            ],
            config: {
                systemInstruction,
                ...generationConfig,
                safetySettings,
            },
        });

        console.log("Generated fine-tuned response successfully.");
        return chatModel.text;
    } catch (error) {
        console.error("Error in handleFineTuned:", error);
    }
}