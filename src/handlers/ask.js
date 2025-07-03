import EmbeddingService from "../services/embeddingService.js";
import PineconeService from "../services/pineconeService.js";
import { CONSTANTS } from "../utils/constants.js";
import { successResponse, errorResponse } from "../utils/response.js";
import { validateQuestion } from "../utils/validation.js";

export const handler = async (event) => {
  try {
    let bodyString = event.body;

    // Decode if base64 encoded
    if (event.isBase64Encoded) {
      bodyString = Buffer.from(event.body, "base64").toString("utf-8");
    }

    const { question, version = CONSTANTS.KNOWLWDGEBASE_VERSION } = JSON.parse(bodyString || "{}");

    validateQuestion(question);

    const embedding = await EmbeddingService.createEmbedding(question);

    const queryRes = await PineconeService.queryVectors(
      embedding,
      undefined,
      5,
      version
    );

    const context = queryRes.map((match) => match.metadata.text).join("\n\n");

    const sourceDocuments = new Map();
    queryRes.forEach((match) => {
      if (match.metadata?.docId && match.metadata?.sourceUrl) {
        sourceDocuments.set(match.metadata.docId, {
          docId: match.metadata.docId,
          sourceUrl: match.metadata.sourceUrl,
          addedAt: match.metadata.addedAt,
        });
      }
    });

    const answer = await EmbeddingService.generateAnswer(context, question);

    return successResponse({
      answer,
      sources: queryRes.map((m) => ({
        score: m.score,
        text: m.metadata.text,
        docId: m.metadata.docId,
        sourceUrl: m.metadata.sourceUrl,
        chunkIndex: m.metadata.chunkIndex,
        addedAt: m.metadata.addedAt,
        version: m.metadata.version, // optional: return version as well
      })),
      sourceDocuments: Array.from(sourceDocuments.values()),
      totalSourceDocuments: sourceDocuments.size,
    });
  } catch (error) {
    console.error("Ask handler error:", error);
    return errorResponse(error);
  }
};
