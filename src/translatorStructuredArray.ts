import { PassThrough } from 'stream';
import { z } from 'zod';
import { JSONParser } from '@streamparser/json-node';
import type { OpenAI } from 'openai';
import log from 'loglevel';

import { TranslationOutput } from './translatorOutput.js';
import { TranslatorStructuredBase } from './translatorStructuredBase.js';
import type { TranslationServiceContext, TranslatorOptions } from './translatorBase.js';

/**
 * Structured translator that uses a JSON array schema (`{ outputs: string[] }`)
 * for translation. Input lines are sent as a JSON array and the model returns
 * translated lines in the same positional order, enabling incremental JSON
 * stream parsing of individual array elements.
 */
export class TranslatorStructuredArray extends TranslatorStructuredBase {
  /**
   * Creates a new array-based structured translator.
   *
   * @param language - Source and target language configuration.
   * @param services - Translation service context (client, cooler, stream callbacks, etc.).
   * @param options - Optional translator configuration overrides.
   */
  constructor(
    language: { from?: string; to: string },
    services: TranslationServiceContext,
    options?: Partial<TranslatorOptions>,
  ) {
    super(language, services, options);
  }

  /**
   * Translates a batch of lines by sending them as a JSON `{ inputs }` array
   * and parsing the structured `{ outputs }` response.
   *
   * @param lines - Array of source text lines to translate.
   * @returns The translation output containing translated lines and usage metadata.
   */
  async doTranslatePrompt(lines: string[]): Promise<TranslationOutput> {
    const userMessage: OpenAI.Chat.ChatCompletionMessageParam = {
      role: 'user',
      content: JSON.stringify({ inputs: lines }),
    };
    const systemMessage: OpenAI.Chat.ChatCompletionMessageParam[] = this.systemInstruction
      ? [{ role: 'system', content: `${this.systemInstruction}` }]
      : [];
    const messages = [...systemMessage, ...this.options.initialPrompts, ...this.promptContext, userMessage];
    const max_tokens = this.getMaxToken(lines as any);

    const structuredArray = z.object({
      outputs: z.array(z.string()),
    });

    try {
      await this.services.cooler?.cool();

      const output = await this.streamParse(
        {
          messages,
          ...this.options.createChatCompletionRequest,
          stream: this.options.createChatCompletionRequest.stream,
          max_tokens,
        } as OpenAI.ChatCompletionCreateParams,
        {
          structure: structuredArray,
          name: 'translation_array',
        },
        true,
      );

      const translationCandidate = output.choices[0].message as any;

      const getLinesOutput = async (translation: any) => {
        if (translation.refusal) {
          return [translation.refusal];
        }
        return translation.parsed.outputs;
      };

      const linesOut = await getLinesOutput(translationCandidate);

      return TranslationOutput.fromCompletion(linesOut, output);
    } catch (error: any) {
      log.error(
        '[TranslatorStructuredArray]',
        `Error ${error?.constructor?.name}`,
        error?.message,
      );
      return this.handleTranslateError(error, lines.length)!;
    }
  }

  /**
   * Formats context lines as JSON for the array structured mode.
   *
   * User-role lines are wrapped as `{ inputs }` and assistant-role lines
   * as `{ outputs }` to match the structured schema format.
   *
   * @param lines - The context lines to format.
   * @param role - Whether these lines represent user input or assistant output.
   * @returns A JSON string representing the context in the appropriate role format.
   */
  override getContextLines(lines: string[], role: 'user' | 'assistant'): string {
    if (role === 'user') {
      return JSON.stringify({ inputs: lines });
    } else {
      return JSON.stringify({ outputs: lines });
    }
  }

  /**
   * Incrementally parses the streamed JSON response to emit each translated
   * array element as it arrives.
   *
   * Pipes the streaming content through a JSONParser targeting `$.outputs.*`
   * and invokes stream callbacks for each completed element, allowing the
   * caller to display results progressively.
   *
   * @param runner - The OpenAI streaming runner instance.
   */
  override jsonStreamParse(runner: any): void {
    this.services.onStreamChunk?.('\n');
    const passThroughStream = new PassThrough();
    let writeBuffer = '';
    runner.on('content.delta', (e: any) => {
      writeBuffer += e.delta;
      passThroughStream.write(e.delta);
      if (writeBuffer) {
        this.services.onStreamChunk?.(writeBuffer);
        writeBuffer = '';
      }
    });

    runner.on('content.done', () => {
      passThroughStream.end();
      this.services.onClearLine?.();
    });

    const pipeline = passThroughStream.pipe(
      new JSONParser({ paths: ['$.outputs.*'], keepStack: false }),
    );

    pipeline.on('data', ({ value: output }: { value: string }) => {
      try {
        this.services.onClearLine?.();
        writeBuffer = `${output}\n`;
      } catch (err) {
        log.error('[TranslatorStructuredArray]', 'Parsing error:', err);
      }
    });

    pipeline.on('error', (err: Error) => {
      log.error('[TranslatorStructuredArray]', 'stream-json parsing error:', err);
    });
  }
}
