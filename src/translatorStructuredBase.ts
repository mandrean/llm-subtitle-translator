import { APIUserAbortError } from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import type { OpenAI } from 'openai';
import type { ZodType } from 'zod';
import log from 'loglevel';
import { Translator } from './translator.js';
import { TranslationOutput } from './translatorOutput.js';
import type { TranslationServiceContext, TranslatorOptions } from './translatorBase.js';

/**
 * Abstract base class for structured-output translators that use Zod schemas
 * and OpenAI's structured response format for type-safe translation results.
 *
 * Disables prefix numbering (incompatible with structured output) and provides
 * shared streaming/parsing infrastructure for all structured translator variants.
 *
 * @template T - The type of a single translated line/entry.
 * @template TLines - The array type for a batch of translated lines/entries.
 */
export abstract class TranslatorStructuredBase<T = string, TLines extends T[] = T[]> extends Translator<T, TLines> {
  /** Backup of the original stream option before any structured-mode overrides. */
  optionsBackup: { stream?: boolean };

  /**
   * Creates a new structured base translator.
   *
   * Forces `prefixNumber` to `false` (required for structured output) and
   * backs up the original stream setting.
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

  /**
   * Handles errors that occur during translation.
   *
   * User-initiated aborts return `undefined`. For multi-line batches, returns an
   * empty output to trigger batch-size reduction. For single-line batches, re-throws
   * the error since no further splitting is possible.
   *
   * @param error - The error thrown during translation.
   * @param lineCount - Number of lines in the batch that failed.
   * @returns An empty translation output for multi-line batches, or `undefined` for aborts.
   * @throws The original error if the batch contains only one line.
   */
  handleTranslateError(error: Error, lineCount: number): TranslationOutput | undefined {
    if (error instanceof APIUserAbortError) {
      return undefined;
    }
    if (lineCount > 1) {
      return new TranslationOutput([] as any, 0, 0, 0, 0);
    }
    throw error;
  }

  /**
   * Sends a chat completion request with Zod-based structured output parsing.
   *
   * In streaming mode, attaches chunk listeners (either raw content deltas or
   * JSON-aware streaming via {@link jsonStreamParse}) and waits for the stream
   * to finish. In non-streaming mode, uses the `parse` API directly.
   *
   * @template ZodInput - The Zod schema type defining the expected response structure.
   * @param params - OpenAI chat completion request parameters.
   * @param zFormat - The Zod schema and schema name for structured response parsing.
   * @param jsonStream - When `true`, delegates to {@link jsonStreamParse} for incremental JSON parsing instead of raw content deltas.
   * @returns The completed chat completion response.
   */
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

  /**
   * Hook for subclasses to implement incremental JSON stream parsing.
   *
   * Called by {@link streamParse} when `jsonStream` is `true`. Subclasses
   * override this to pipe streaming content through a JSON parser and emit
   * individual translated entries as they arrive.
   *
   * @param runner - The OpenAI streaming runner instance.
   */
  jsonStreamParse(runner: any): void {
    // Abstract - subclasses override
  }
}
