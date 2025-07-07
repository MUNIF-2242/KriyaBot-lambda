import PDFService from "../services/pdfService.js";
import PineconeService from "../services/pineconeService.js";
import EmbeddingService from "../services/embeddingService.js";

export const handler = async (event) => {
  for (const record of event.Records) {
    try {
      const { url } = JSON.parse(record.body);

      if (!url) throw new Error("Missing URL");

      await PineconeService.ensureIndexExists();

      const docId = PDFService.generateDocumentId(url);
      const exists = await PineconeService.documentExists(docId);

      if (exists) {
        console.log("Document already indexed:", docId);
        continue;
      }

      const filePath = await PDFService.downloadPDF(url, docId);
      const text = await PDFService.extractTextFromPDF(filePath);
      await PDFService.cleanupFile(filePath);

      const chunks = PDFService.splitText(text);
      console.log("Number of chunks:", chunks.length);

      const embeddings = await EmbeddingService.createEmbeddings(chunks);

      const addedAt = new Date().toISOString();
      const version = "v1";

      const vectors = embeddings.map((embedding, i) => ({
        id: `${docId}-chunk-${i}`,
        values: embedding,
        metadata: {
          text: chunks[i],
          docId,
          sourceUrl: url,
          chunkIndex: i,
          addedAt,
          version,
        },
      }));

      await PineconeService.upsertVectors(vectors);

      console.log("Successfully indexed:", docId);
    } catch (error) {
      console.error("Error processing SQS message:", error);
    }
  }
};
