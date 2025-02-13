import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatPromptTemplate,MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

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

export function createDocRetrievalChain(retriever){
    const convertDocsToString = (documents) => {
        return documents.map((doc) => {
            return `<doc>\n${doc.pageContent}\n</doc>`
        }).join("\n");
    }
    
    const docRetrieverChain = RunnableSequence.from([
        (input) => input.question,
        retriever,
        convertDocsToString
    ])

    return docRetrieverChain;
}

export function createRephraseQuestionChain(){
    const REPHRASE_QUESTION_SYSTEM_TEMPLATE =
        `Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question.`;
    
    const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
        ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
        new MessagesPlaceholder("history"),
        ["human", "Rephrase the following question as a standalone question:\n{question}"]
    ])
    
    const rephraseQuestionChain = RunnableSequence.from([
        rephraseQuestionChainPrompt,
        new ChatOpenAI({
            apiKey: process.env.GITHUB_TOKEN,
            temperature: 0.1,
            configuration: {
                baseURL: "https://models.inference.ai.azure.com"
            },
            model: "gpt-4o"
        }),
        new StringOutputParser(),
    ])

    return rephraseQuestionChain;
}