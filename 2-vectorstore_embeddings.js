import dotenv from "dotenv";
dotenv.config();
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings(
    {
        apiKey: process.env.GITHUB_TOKEN,
        configuration: {
            baseURL: "https://models.inference.ai.azure.com"
        },
        model: "text-embedding-3-small"
    }
);
const embedded = await embeddings.embedQuery("Hello I'm Atanu, How are you?")
console.log(embedded);
