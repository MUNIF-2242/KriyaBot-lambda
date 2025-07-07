import { SQSClient, SendMessageCommand } from "@aws-sdk/client-sqs";
import { successResponse, errorResponse } from "../utils/response.js";

const sqs = new SQSClient({ region: process.env.AWS_REGION });

export const handler = async (event) => {
  try {
    console.log("📥 Raw event:", JSON.stringify(event, null, 2));

    // 🛡️ Robust body parsing
    let body;
    if (event.isBase64Encoded) {
      const decoded = Buffer.from(event.body, "base64").toString("utf-8");
      body = JSON.parse(decoded);
    } else if (typeof event.body === "string") {
      body = JSON.parse(event.body);
    } else if (typeof event.body === "object" && event.body !== null) {
      body = event.body;
    } else {
      throw new Error("Invalid request body");
    }

    const { url } = body;

    if (!url || typeof url !== "string") {
      throw new Error("Invalid or missing 'url'");
    }

    const command = new SendMessageCommand({
      QueueUrl: process.env.PDF_PROCESSING_QUEUE_URL,
      MessageBody: JSON.stringify({ url }),
    });

    await sqs.send(command);

    return successResponse(
      { message: "✅ PDF processing queued successfully" },
      202
    );
  } catch (err) {
    console.error("❌ Queue error:", err);
    return errorResponse(err);
  }
};
