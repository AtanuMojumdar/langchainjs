import dotenv from "dotenv";
dotenv.config();
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const embeddings = new OpenAIEmbeddings(
    {
        apiKey: process.env.GITHUB_TOKEN,
        configuration: {
            baseURL: "https://models.inference.ai.azure.com"
        },
        model: "text-embedding-3-small"
    }
);


const loader = new PDFLoader("pdf/machinelearning-lecture01.pdf");

const rawData = await loader.load();

const textsplitter = new RecursiveCharacterTextSplitter(
    {
        chunkSize:128,
        chunkOverlap:0,
    }
)

const splitDocs = await textsplitter.splitDocuments(rawData);

//Vectore Store

const store = new MemoryVectorStore(embeddings);
const re = await store.addDocuments(splitDocs);

//search using NL
const retrievedocs = await store.similaritySearch(
    "what is deep learning",
)
console.log(retrievedocs)