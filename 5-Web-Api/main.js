import dotenv from "dotenv";
import path from "path"
import { fileURLToPath } from "url";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({
    path: path.join(__dirname, "..", ".env")
});
import { loaderAndSplitter, initVectorStoreWithDocs, createDocRetrievalChain, createRephraseQuestionChain } from "./helpers.js";
import { RunnableSequence, RunnablePassthrough, RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HttpResponseOutputParser } from "langchain/output_parsers";
import { ChatMessageHistory } from "langchain/memory";
import express from "express";

const app = express()


const splitDocs = await loaderAndSplitter(128, 1536);

const vectorStore = await initVectorStoreWithDocs(splitDocs);
const retriever = vectorStore.asRetriever();


const docRetrievalChain = createDocRetrievalChain(retriever);
const rephraseQuestionChain = createRephraseQuestionChain();

//Putting all it together!
const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history, 
answer the user's question to the best of 
your ability 
using only the resources provided. Be verbose!

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    [
        "human",
        "Now, answer this question using the previous context and chat history:\n{standalone_question}"
    ]
]);

const conversationalRetrievalChain = RunnableSequence.from([
    RunnablePassthrough.assign({ // history + question + standalone_question
        standalone_question: rephraseQuestionChain
    }),
    RunnablePassthrough.assign({ // history + question + standalone_question
        context: docRetrievalChain
    }),
    answerGenerationChainPrompt,
    new ChatOpenAI({
        apiKey: process.env.GITHUB_TOKEN,
        configuration: {
            baseURL: "https://models.inference.ai.azure.com"
        },
        model: "gpt-4o"
    })
])

const httpResponseOutputParser = new HttpResponseOutputParser({
    contentType: "text/plain"
})

const messageHistory = new ChatMessageHistory();

// const finalRetrievalChain = new RunnableWithMessageHistory({
//     runnable: conversationalRetrievalChain,
//     getMessageHistory: (_sessionId)=> messageHistory,
//     historyMessagesKey: "history",
//     inputMessagesKey: "question"
// }).pipe(httpResponseOutputParser);

const messageHistories = {};

const getMessageHistoryForSession = (sessionId) => {
    if (messageHistories[sessionId] !== undefined) {
        return messageHistories[sessionId];
    }
    const newChatSessionHistory = new ChatMessageHistory();
    messageHistories[sessionId] = newChatSessionHistory;
    return newChatSessionHistory;
};

const finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: getMessageHistoryForSession,
    inputMessagesKey: "question",
    historyMessagesKey: "history",
}).pipe(httpResponseOutputParser);

app.use(express.json());

app.get("/", async (req, res) => {
    try {
        const { question, sessionId } = req.body;
        const stream = await finalRetrievalChain.stream({
            question: question,
        }, {
            configurable: {
                sessionId: sessionId,
            }
        })
        res.setHeader('Content-Type', 'text/plain');
        res.status(200);
        for await (const chunk of stream) {
            res.write(chunk);
        }

        res.end();
    }
    catch (err) {
        console.log(err)
        return res.send("Error!")
    }

})

app.listen(8000, () => {
    console.log("Listening")
})