import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export async function loaderAndSplitter(chunkOverlap, chunkSize) {
    const loader = new PDFLoader("pdf/machinelearning-lecture01.pdf")
    const rawSplitDocs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkOverlap: chunkOverlap,
        chunkSize: chunkSize,
    })

    const splitDocs = await splitter.splitDocuments(rawSplitDocs);
    return splitDocs;
}

export async function initVectorStoreWithDocs(documents) {
    const embeddingModel = new OpenAIEmbeddings({
        apiKey: process.env.GITHUB_TOKEN,
        configuration:{
            baseURL: "https://models.inference.ai.azure.com",
        },
        model: "text-embedding-3-small",
    })

    const vectorStore = new MemoryVectorStore(embeddingModel);
    await vectorStore.addDocuments(documents);
    return vectorStore;
}

