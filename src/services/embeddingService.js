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
    const prompt = `You are Kriyakarak, a friendly and helpful virtual assistant created to support users of the Kriyakarak platform.

Only answer questions using the information in the context below. If the context includes the answer, respond naturally and clearly—summarize the meaning without quoting or repeating it word-for-word.

If you cannot find the answer in the context, do not attempt to answer it. Instead, guide the user to contact the Kriyakarak support team for help, and end with a friendly question offering further help.

If the user is greeting you or making small talk (like "hello" or "how are you"), respond warmly and conversationally, just like a human would.

Whenever helpful, provide useful [links](https://kriyakarak.com/contact) or direct users to where they can find more help on the platform.

**LINK FORMATTING:**
- ALWAYS provide clickable links when available in the context using this exact format: [Link Text](URL)
- Create meaningful, descriptive link text that tells users what they'll find when they click
- Examples of good link formatting:
  - [Visit our Help Center](https://kriyakarak.com/contact) 
  - [Contact Support Team](https://kriyakarak.com/contact)
  - [View Tutorial Guide](https://www.youtube.com/@KriyaKarak)

Avoid technical language, citations, or any mention of “context.” Be concise, friendly, and supportive.

If the answer isn’t found in the context, respond with:
"I'm not sure about that at the moment. Please contact our support team at 📞 +8801712651400  or [visit our Help Center](https://kriyakarak.com/contact) for further help.  
”

---
Context:
${context}

Question: ${question}

Answer:`;





    const completion = await this.openai.chat.completions.create({
      model: this.chatModel,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.2,
      // top_p: 1,
      max_tokens: 200,
    });

    return completion.choices[0].message.content;
  }
}

const EmbeddingService = new EmbeddingServiceClass();
export default EmbeddingService;
