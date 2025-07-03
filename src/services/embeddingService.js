import { config } from "dotenv";
config(); // Load environment variables

import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";

class EmbeddingServiceClass {
  constructor() {
    // Bedrock setup for Titan Embedding and LLaMA 3
    const awsRegion = process.env.AWS_REGION || "us-east-1";
    this.bedrockClient = new BedrockRuntimeClient({
      region: awsRegion,
    });

    // Model IDs
    this.embeddingModelId = "amazon.titan-embed-text-v1";
    this.llamaModelId = "meta.llama3-70b-instruct-v1:0";
  }

  // Create embedding for single text using Titan
  async createEmbedding(text) {
    const body = {
      inputText: text,
    };

    const command = new InvokeModelCommand({
      modelId: this.embeddingModelId,
      body: JSON.stringify(body),
      contentType: "application/json",
      accept: "application/json",
    });

    try {
      const response = await this.bedrockClient.send(command);
      const responseBody = await response.body.transformToString();
      const result = JSON.parse(responseBody);
      return result.embedding;
    } catch (error) {
      console.error("Error from Titan Embedding:", error);
      return null;
    }
  }

  // Create embeddings for an array of texts
  async createEmbeddings(texts) {
    const embeddings = await Promise.all(
      texts.map((text) => this.createEmbedding(text))
    );
    return embeddings;
  }

  // Format prompt with LLaMA 3 special tokens
  formatPrompt(userPrompt) {
    return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n${userPrompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n`;
  }

  // Generate answer from LLaMA 3 via Bedrock
  async generateAnswer(context, question) {
    const userPrompt = `You are Kriyakarak, a helpful and friendly assistant for users of the Kriyakarak platform.

Your task is to answer user questions using ONLY the most recent information provided in the "Latest Knowledge" section below. Do not use any external knowledge or assumptions.

When relevant information is available:
- Answer in a natural, conversational tone.
- Summarize or explain key points clearly and helpfully.
- Use clickable [links](https://kriyakarak.com/contact) following markdown rules.

When the information is incomplete or missing:
- Gently let the user know you're not sure.
- Offer to connect them with the support team.
- Stay warm and polite.

For greetings or small talk:
- Respond like a friendly human assistant would.

**LINK RULES**
- Use markdown links like: [Contact Support](https://kriyakarak.com/contact)
- Examples:
  - [Visit the Help Center](https://kriyakarak.com/contact)
  - [Contact Support Team](https://kriyakarak.com/contact)
  - [Watch a Tutorial](https://www.youtube.com/@KriyaKarak)

**DO NOT**
- Mention the word "context"
- Repeat the question
- Use technical or robotic language
- Make up or hallucinate any information

---
Latest Knowledge:
${context}

Question:
${question}

Answer:`;

    const formattedPrompt = this.formatPrompt(userPrompt);

    const body = {
      prompt: formattedPrompt,
      temperature: 0.2,
      top_p: 0.9,
      max_gen_len: 128,
    };

    const command = new InvokeModelCommand({
      modelId: this.llamaModelId,
      body: JSON.stringify(body),
      contentType: "application/json",
      accept: "application/json",
    });

    try {
      const response = await this.bedrockClient.send(command);
      const responseBody = await response.body.transformToString();
      const result = JSON.parse(responseBody);
      return result.generation || "No response generated.";
    } catch (error) {
      console.error("Error from Bedrock (LLaMA 3):", error);
      return "Sorry, I couldn't process your request at the moment.";
    }
  }
}

const EmbeddingService = new EmbeddingServiceClass();
export default EmbeddingService;
