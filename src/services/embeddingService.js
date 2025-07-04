import { config } from "dotenv";
config(); // Load environment variables

import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";

class EmbeddingServiceClass {
  constructor() {
    const awsRegion = process.env.AWS_REGION || "us-east-1";
    this.bedrockClient = new BedrockRuntimeClient({ region: awsRegion });

    this.embeddingModelId = "amazon.titan-embed-text-v1";
    this.llamaModelId = "meta.llama3-70b-instruct-v1:0";
  }

  async createEmbedding(text) {
    const command = new InvokeModelCommand({
      modelId: this.embeddingModelId,
      body: JSON.stringify({ inputText: text }),
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

  async createEmbeddings(texts) {
    return Promise.all(texts.map((text) => this.createEmbedding(text)));
  }

  formatPrompt(userPrompt) {
    return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n${userPrompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n`;
  }

  async generateAnswer(context, question) {
    const userPrompt = `You are Kriyakarak, a helpful and friendly assistant for users of the Kriyakarak platform.

Always answer using ONLY the most recent information provided in the "Latest Knowledge" section. Ignore any outdated or unrelated data.

**Give short, precise, and meaningful answers** (1–3 sentences max).

✅ If the answer is in the latest knowledge:
- Respond naturally and directly.
- Do not explain too much.
- Add helpful [links](https://kriyakarak.com/contact) when appropriate.

⚠️ If the answer is missing:
- Politely say you're not sure.
- Offer to [Contact Support](https://kriyakarak.com/contact)

❌ DO NOT:
- Mention "context"
- Repeat the question
- Use robotic or technical tone
- Make up anything

---
Latest Knowledge (always use the most recent data below):
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
