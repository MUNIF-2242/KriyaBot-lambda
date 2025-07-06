import { config } from "dotenv";
config(); // Load environment variables

import {
  BedrockRuntimeClient,
  ConverseStreamCommand,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";

class EmbeddingServiceClass {
  constructor() {
    const awsRegion = process.env.REGION_FOR_BEDROCK || "us-east-1";
    this.bedrockClient = new BedrockRuntimeClient({ region: awsRegion });

    this.embeddingModelId = "amazon.titan-embed-text-v1";
    this.claudeModelId = "anthropic.claude-3-haiku-20240307-v1:0";
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

  async generateAnswer(context, question) {
    const userPrompt = `You are Kriyakarak, a helpful and friendly assistant for users of the Kriyakarak platform.

**
Always answer using ONLY the most recent information provided in the "Latest Knowledge" section. Ignore any outdated or unrelated data. If the context includes an answer, give a natural, brief response that reflects the meaning without repeating or quoting it directly.

If the answer is not in the context but is something you can answer from general knowledge, just answer it naturally—**do not mention context or general knowledge explicitly**, and avoid using parentheses.

If the question is a greeting or small talk (like "hi", "how are you"), respond conversationally, as a human would.

Be warm, clear, and concise. Avoid technical phrasing, citations, or brackets.** 

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
Latest Knowledge:
${context}

Question:
${question}`;

    const messages = [
      {
        role: "user",
        content: [{ text: userPrompt }],
      },
    ];

    const command = new ConverseStreamCommand({
      modelId: this.claudeModelId,
      messages,
      inferenceConfig: {
        temperature: 0.2,
        topP: 0.9,
        maxTokens: 512,
      },
    });

    try {
      const response = await this.bedrockClient.send(command);

      let finalAnswer = "";
      for await (const item of response.stream) {
        if (item.contentBlockDelta?.delta?.text) {
          finalAnswer += item.contentBlockDelta.delta.text;
        }
      }

      return finalAnswer || "No response generated.";
    } catch (error) {
      console.error("Error from Bedrock Claude Haiku:", error);
      return "Sorry, I couldn't process your request at the moment.";
    }
  }
}

const EmbeddingService = new EmbeddingServiceClass();
export default EmbeddingService;
