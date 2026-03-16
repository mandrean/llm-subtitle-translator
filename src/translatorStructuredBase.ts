import { APIUserAbortError } from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import type { OpenAI } from 'openai';
import type { ZodType } from 'zod';
import log from 'loglevel';
import { Translator } from './translator.js';
import { TranslationOutput } from './translatorOutput.js';
import type { TranslationServiceContext, TranslatorOptions } from './translatorBase.js';

export abstract class TranslatorStructuredBase<T = string, TLines extends T[] = T[]> extends Translator<T, TLines> {
  optionsBackup: { stream?: boolean };

  constructor(
    language: { from?: string; to: string },
    services: TranslationServiceContext,
    options?: Partial<TranslatorOptions>,
  ) {
    const opts = options ?? {};
    log.debug(`[TranslatorStructuredBase]`, 'Structured Mode:', opts.structuredMode);
    const optionsBackup: { stream?: boolean } = {};
    optionsBackup.stream = opts.createChatCompletionRequest?.stream ?? undefined;
    if (opts.prefixNumber) {
      log.warn(
        '[TranslatorStructuredBase]',
        '--no-prefix-number must be used in structured mode, overriding.',
      );
    }
    opts.prefixNumber = false;
    super(language, services, opts);

    this.optionsBackup = optionsBackup;
  }

  handleTranslateError(error: Error, lineCount: number): TranslationOutput | undefined {
    if (error instanceof APIUserAbortError) {
      return undefined;
    }
    if (lineCount > 1) {
      return new TranslationOutput([] as any, 0, 0, 0, 0);
    }
    throw error;
  }

  async streamParse<ZodInput extends ZodType>(
    params: OpenAI.ChatCompletionCreateParams,
    zFormat: { structure: ZodInput; name: string },
    jsonStream: boolean = false,
  ): Promise<OpenAI.Chat.ChatCompletion> {
    const zodResponseFormatOutput = zodResponseFormat(zFormat.structure, zFormat.name);
    if (params.stream) {
      const runner = this.client.chat.completions.stream({
        ...params,
        response_format: zodResponseFormatOutput,
        stream: true,
        stream_options: {
          include_usage: true,
        },
      });

      this.streamController = runner.controller;

      if (jsonStream) {
        this.jsonStreamParse(runner);
      } else {
        runner.on('content.delta', (e) => {
          this.services.onStreamChunk?.(e.delta);
        });
      }
      await runner.done();

      this.services.onStreamEnd?.();

      const final = await runner.finalChatCompletion();

      return final as any;
    } else {
      const output = await this.client.chat.completions.parse({
        ...params,
        response_format: zodResponseFormatOutput,
        stream: false,
      });
      return output as any;
    }
  }

  jsonStreamParse(runner: any): void {
    // Abstract - subclasses override
  }
}
