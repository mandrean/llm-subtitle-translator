import { PassThrough } from 'stream';
import { z } from 'zod';
import { JSONParser } from '@streamparser/json-node';
import type { OpenAI } from 'openai';
import log from 'loglevel';

import { TranslationOutput } from './translatorOutput.js';
import { TranslatorStructuredBase } from './translatorStructuredBase.js';
import type { TranslationServiceContext, TranslatorOptions } from './translatorBase.js';

export class TranslatorStructuredArray extends TranslatorStructuredBase {
  constructor(
    language: { from?: string; to: string },
    services: TranslationServiceContext,
    options?: Partial<TranslatorOptions>,
  ) {
    super(language, services, options);
  }

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

  override getContextLines(lines: string[], role: 'user' | 'assistant'): string {
    if (role === 'user') {
      return JSON.stringify({ inputs: lines });
    } else {
      return JSON.stringify({ outputs: lines });
    }
  }

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
