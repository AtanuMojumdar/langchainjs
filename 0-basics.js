import dotenv from 'dotenv'
dotenv.config();
import { ChatOpenAI } from '@langchain/openai';
// import { HumanMessage } from "@langchain/core/messages"
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';


const model = new ChatOpenAI({
    apiKey: process.env.GITHUB_TOKEN,
    configuration: {
        baseURL: "https://models.inference.ai.azure.com"
    },
    model: "gpt-4o"
})

// model.invoke([new HumanMessage("Hello I'm Atanu")])
// .then((val)=>{
//     console.log(val);

// })
//----------------------------

// Prompt Template
// const prompt = ChatPromptTemplate.fromTemplate(`Three good names for company making this {product}`);

// prompt.formatMessages({
//     product: "pencil"
// }).then((val)=> console.log(val))

//-----------------------------
const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a god of war'],
    ['human', '{what} challenges you']
])

//Chain
// const chain = prompt.pipe(model);

// chain.invoke({
//     what: "Atanu"
// }).then((val) => console.log(val))


const outputParser = new StringOutputParser();
// const newChain = prompt.pipe(model).pipe(outputParser);
// newChain.invoke({
//     what: "Atanu"
// }).then((val)=> console.log(val))

const runnableChain = RunnableSequence.from(
    [
        prompt,
        model,
        outputParser
    ]
)

// runnableChain.invoke({
//     what: "Zed"
// }).then((val)=>console.log(val))

const res_stream = await runnableChain.stream({
    what: "Yasuo"
})

for await (const chunk of res_stream){
    console.log(chunk)
}