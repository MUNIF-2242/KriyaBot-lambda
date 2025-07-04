import EmbeddingService from "../services/embeddingService.js";
import PineconeService from "../services/pineconeService.js";
import { successResponse, errorResponse } from "../utils/response.js";
import { validateQuestion } from "../utils/validation.js";

export const handler = async (event) => {
  try {
    // Parse the request body
    let bodyString = event.body;

    if (event.isBase64Encoded) {
      bodyString = Buffer.from(event.body, "base64").toString("utf-8");
    }

    const { userQuestion } = JSON.parse(bodyString || "{}");

    // Validate input
    validateQuestion(userQuestion);

    // Create embedding from the question
    const embedding = await EmbeddingService.createEmbedding(userQuestion);

    // Query Pinecone with the embedding
    const queryRes = await PineconeService.queryVectors(
      embedding,
      undefined,
      5
    );

    if (!queryRes || queryRes.length === 0) {
      throw new Error("No relevant context found.");
    }

    // Sort results to get the most recent context
    const sortedMatches = queryRes.sort(
      (a, b) => new Date(b.metadata.addedAt) - new Date(a.metadata.addedAt)
    );

    const latestMatch = sortedMatches[0];
    const context = `(${latestMatch.metadata.addedAt}) ${latestMatch.metadata.text}`;

    console.log("################ CONTEXT ################");
    console.log(context);

    // Generate answer based on the latest context
    const assistentAnswer = await EmbeddingService.generateAnswer(
      context,
      userQuestion
    );

    return successResponse({
      userQuestion,
      assistentAnswer,
      source: {
        id: latestMatch.id,
        docId: latestMatch.metadata.docId,
        chunkIndex: latestMatch.metadata.chunkIndex,
        sourceUrl: latestMatch.metadata.sourceUrl,
        originalText: latestMatch.metadata.text,
        addedAt: latestMatch.metadata.addedAt,
      },
    });
  } catch (error) {
    console.error("Ask handler error:", error);
    return errorResponse(error);
  }
};
