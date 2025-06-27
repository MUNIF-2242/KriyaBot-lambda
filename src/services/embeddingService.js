import { OpenAI } from "openai";

class EmbeddingServiceClass {
  constructor(apiKey = process.env.OPENAI_API_KEY) {
    if (!apiKey) {
      throw new Error("Missing OpenAI API key");
    }

    this.openai = new OpenAI({ apiKey });
    this.embeddingModel = "text-embedding-3-small";
    this.chatModel = "gpt-4.1-nano";
  }

  async createEmbedding(text) {
    const response = await this.openai.embeddings.create({
      model: this.embeddingModel,
      input: text,
    });
    return response.data[0].embedding;
  }

  async createEmbeddings(texts) {
    const embeddings = await Promise.all(
      texts.map((text) => this.createEmbedding(text))
    );
    return embeddings;
  }

 async generateAnswer(context, question) {
  const prompt = `You are Kriyakarak, a helpful and friendly assistant for users of the Kriyakarak platform.

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

  const completion = await this.openai.chat.completions.create({
    model: this.chatModel,
    messages: [{ role: "user", content: prompt }],
    temperature: 0.2,
    max_tokens: 200,
  });

  return completion.choices[0].message.content;
}

}

const EmbeddingService = new EmbeddingServiceClass();
export default EmbeddingService;
