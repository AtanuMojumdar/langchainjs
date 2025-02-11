import dotenv from "dotenv";
import path from "path"
import { fileURLToPath } from "url";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({
    path: path.join(__dirname, "..", ".env")
});
import { loaderAndSplitter, initVectorStoreWithDocs } from "./helpers.js";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableMap } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";


const splitDocs = await loaderAndSplitter(128, 1536);

const vectorStore = await initVectorStoreWithDocs(splitDocs);
const retriever = vectorStore.asRetriever();

//Document retriever chain
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

// const result = await docRetrieverChain.invoke({
//     question: "What are prerequisites of this course?"
// })

// console.log(result)

//Synthesizing a response
const TEMPLATE = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`

const answerGenPrompt = ChatPromptTemplate.fromTemplate(TEMPLATE);

const runnableMap = RunnableMap.from({ //calls runnables and functions with same input
    context: docRetrieverChain,
    question: (input) => input.question
})

// const res = await runnableMap.invoke({
//     question: "What are the prerequisites of this course?"
// })
// console.log(res)


//Augmented Generation

const model = new ChatOpenAI({
    apiKey: process.env.GITHUB_TOKEN,
    configuration: {
        baseURL: "https://models.inference.ai.azure.com"
    },
    model: "gpt-4o"
})

const outputParser = new StringOutputParser();

const runnableChain = RunnableSequence.from([
    {
        context: docRetrieverChain,
        question: (input) => input.question
    },
    answerGenPrompt,
    model,
    outputParser,
])

const res = await runnableChain.invoke({
    question: "What are the prerequisites of this course?"
});

console.log(res)