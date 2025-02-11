import dotenv from "dotenv";
import path from "path"
import { fileURLToPath } from "url";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({
    path: path.join(__dirname, "..", ".env")
});
import { loaderAndSplitter, initVectorStoreWithDocs } from "./helpers.js";
import { RunnableSequence, RunnablePassthrough, RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { ChatMessageHistory } from "langchain/memory";


const splitDocs = await loaderAndSplitter(128, 1536);
const vectorStore = await initVectorStoreWithDocs(splitDocs);
const retriever = vectorStore.asRetriever();

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

const model = new ChatOpenAI({
    apiKey: process.env.GITHUB_TOKEN,
    configuration: {
        baseURL: "https://models.inference.ai.azure.com"
    },
    model: "gpt-4o"
})


const runnableChain = RunnableSequence.from([
    {
        context: docRetrieverChain,
        question: (input) => input.question
    },
    answerGenPrompt,
    model,
    new StringOutputParser(),
])


//Adding History 

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

const originalQuestion = "What are prerequisites of this course?";

// const originalAnswer = await runnableChain.invoke({
//     question: originalQuestion
// })

// console.log(originalAnswer)

// const chatHistory = [
//     new HumanMessage(originalQuestion),
//     new AIMessage(originalAnswer),
// ]

// const response = await rephraseQuestionChain.invoke({
//     question: "Can you list them in paragraph form ?",
//     history: chatHistory,
// })
// console.log(response)



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
        context: docRetrieverChain
    }),
    answerGenerationChainPrompt,
    model,
    new StringOutputParser(),
])

const messageHistory = new ChatMessageHistory();

const finalRetrievalChain = new RunnableWithMessageHistory({
    runnable: conversationalRetrievalChain,
    getMessageHistory: (_sessionId)=> messageHistory,
    historyMessagesKey: "history",
    inputMessagesKey: "question"
})

const preFinalRes = await finalRetrievalChain.invoke({
    question: originalQuestion
},{
    configurable: {
        sessionId: "test"
    }
})

const finalRes = await finalRetrievalChain.invoke({
    question: "Can you list them in paragraph form ?"
},{
    configurable: {
        sessionId: "test"
    }
})

console.log(finalRes)