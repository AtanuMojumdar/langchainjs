import { GithubRepoLoader } from "@langchain/community/document_loaders/web/github"; 
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";


//Data Load
const loader = new GithubRepoLoader("https://github.com/langchain-ai/langchainjs",{
    recursive: false,
    ignorePaths: ["*.md","yarn.lock"],
})

const docs = await loader.load();

//Splitting
const splitter = RecursiveCharacterTextSplitter.fromLanguage("js",{
    chunkSize: 32,
    chunkOverlap: 0,
})

const code = `function helloWorld(){
console.log("Hello!");
}
//Call
helloWorld();
`
const data = await splitter.splitText(code);
// const data = await splitter.splitDocuments(code);
console.log(data);

