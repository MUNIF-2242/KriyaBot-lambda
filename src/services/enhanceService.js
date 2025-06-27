// services/enhanceService.js
import { OpenAI } from "openai";

class EnhanceServiceClass {
  constructor(apiKey = process.env.OPENAI_API_KEY) {
    if (!apiKey) {
      throw new Error("Missing OpenAI API key");
    }

    this.openai = new OpenAI({ apiKey });
    this.model = "gpt-4.1-nano"; // or gpt-3.5-turbo for cheaper option
  }

  async enhanceText(text) {
    const prompt = `Please enhance the following text. Rephrase it to be clearer, more professional, and user-friendly without changing its meaning.

---
Original Text:
"${text}"

Enhanced Version:
`;

    const completion = await this.openai.chat.completions.create({
      model: this.model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.4,
      max_tokens: 400,
    });

    return completion.choices[0].message.content;
  }
}

const EnhanceService = new EnhanceServiceClass();
export default EnhanceService;
