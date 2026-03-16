import { OpenAI } from 'openai';
import type { Stream } from 'openai/streaming';
import log from 'loglevel';
import { retryWrapper, sleep } from './helpers.js';

export class ChatStreamSyntaxError extends SyntaxError {
  constructor(message: string, cause?: ErrorOptions) {
    super(message, cause);
  }
}

export async function openaiRetryWrapper<T>(
  func: () => Promise<T>,
  maxRetries: number,
  description: string,
): Promise<T | undefined> {
  return await retryWrapper(
    func,
    maxRetries,
    async (retryContext) => {
      const error = retryContext.error;
      let delay = 1000 * retryContext.currentTry * retryContext.currentTry;
      if (error instanceof OpenAI.APIError) {
        log.error(
          `[Error_${description}]`,
          new Date(),
          'Status',
          error.status,
          error.name,
          error.message,
          error.error,
        );

        if (error.status === 429 || (error.status! >= 500 && error.status! <= 599)) {
          delay = delay * retryContext.currentTry;
        } else {
          throw `[Error_${description}] ${new Date()} ${error.message}`;
        }
        log.error(`[Error_${description}]`, 'Retries', retryContext.currentTry, 'Delay', delay);
        await sleep(delay);
      } else if (error instanceof ChatStreamSyntaxError) {
        log.error(
          `[Error_${description}] ${error.message}`,
          'Retries',
          retryContext.currentTry,
          'Delay',
          delay,
        );
        await sleep(delay);
      } else {
        throw `[Error_${description}] [openaiRetryWrapper] ${new Date()} unknown error ${error}`;
      }
    },
    async (retryContext) => {
      log.error(
        `[Error_${description}] [openaiRetryWrapper] Max Retries Reached`,
        new Date(),
        retryContext,
      );
      throw `[Error_${description}] [openaiRetryWrapper] Max Retries Reached, Error: ${retryContext.error?.message ?? retryContext.error}`;
    },
  );
}

export async function completeChatStream(
  response: Stream<OpenAI.Chat.Completions.ChatCompletionChunk>,
  onData: (d: string) => void = () => {},
  onEnd: (u: OpenAI.Completions.CompletionUsage | undefined) => void = () => {},
): Promise<string> {
  let output = '';
  return await new Promise<string>(async (resolve, reject) => {
    try {
      let usage: OpenAI.Completions.CompletionUsage | undefined;
      for await (const part of response) {
        const text = part.choices[0]?.delta?.content;
        if (text) {
          output += text;
          onData(text);
        } else if (part.usage) {
          usage = part.usage;
        }
      }
      onEnd(usage);
      resolve(output);
    } catch (error: any) {
      const chatStreamError = new ChatStreamSyntaxError(
        `Could not JSON parse stream message: ${error.message}`,
        error,
      );
      reject(chatStreamError);
    }
  });
}
