import { z } from 'zod';
import type { OpenAI } from 'openai';
import log from 'loglevel';

import { TranslationOutput } from './translatorOutput.js';
import { TranslatorStructuredBase } from './translatorStructuredBase.js';
import type { TranslationServiceContext, TranslatorOptions } from './translatorBase.js';

/**
 * Key prefix used to identify nested object entries in the Zod schema.
 * Lines containing `\\N` (ASS/SSA multiline markers) are split into nested
 * sub-objects whose schema key starts with this prefix followed by the line index.
 */
const NestedPlaceholder = 'nested_';

/**
 * Structured translator that uses a dynamic JSON object schema where each
 * source line becomes a named property. The model returns an object with the
 * same keys mapped to their translations, preserving positional correspondence.
 *
 * Supports multiline entries (containing `\\N`) by nesting them as sub-objects.
 * Batch sizes are capped more conservatively because each line becomes a
 * distinct schema property.
 */
export class TranslatorStructuredObject extends TranslatorStructuredBase {
  /**
   * Creates a new object-based structured translator.
   *
   * Automatically reduces default batch sizes to `[10, 20]` and rejects
   * batch sizes exceeding 100 to stay within schema property limits.
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
    if (opts.batchSizes?.[0] === 10 && opts.batchSizes?.[1] === 50) {
      const reducedBatchSizes = [10, 20];
      log.warn(
        '[TranslatorStructuredObject]',
        '--batch-sizes is to be reduced to',
        JSON.stringify(reducedBatchSizes),
      );
      opts.batchSizes = reducedBatchSizes;
    } else if (opts.batchSizes?.some((x) => x > 100)) {
      throw new Error('[TranslatorStructuredObject] Batch sizes should not exceed 100');
    }

    super(language, services, opts);
  }

  /**
   * Translates a batch of lines by building a dynamic Zod object schema where
   * each source line text is a property key.
   *
   * Lines containing `\\N` are split into nested sub-objects. The model returns
   * an object with the same keys mapped to translated values, which are then
   * reassembled into a flat output array (with nested values rejoined by `\\N`).
   *
   * @param lines - Array of source text lines to translate.
   * @returns The translation output containing translated lines and usage metadata.
   */
  async doTranslatePrompt(lines: string[]): Promise<TranslationOutput> {
    const systemMessage: OpenAI.Chat.ChatCompletionMessageParam[] = this.systemInstruction
      ? [{ role: 'system', content: `${this.systemInstruction}` }]
      : [];
    const messages = [...systemMessage, ...this.options.initialPrompts, ...this.promptContext];
    const max_tokens = this.getMaxToken(lines as any);

    const structuredObject: Record<string, any> = {};
    for (const [key, value] of (lines as string[]).entries()) {
      if (value.includes('\\N')) {
        const nestedObject: Record<string, any> = {};
        for (const [_nestedKey, nestedValue] of value.split('\\N').entries()) {
          nestedObject[nestedValue.replaceAll('\\', '').trim()] = z.string();
        }
        structuredObject[NestedPlaceholder + key] = z.object({ ...nestedObject });
      } else {
        structuredObject[value.replaceAll('\\', '')] = z.string();
      }
    }
    const translationBatch = z.object({ ...structuredObject });

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
          structure: translationBatch,
          name: 'translation_object',
        },
      );

      const translation = output.choices[0].message as any;

      function getLinesOutput() {
        if (translation.refusal) {
          return [translation.refusal];
        } else {
          const parsed = output.choices[0].message as any;
          const parsedData = parsed.parsed;
          const linesOut: string[] = [];

          let expectedIndex = 0;
          for (const [key, value] of Object.entries(parsedData)) {
            if (key.startsWith(NestedPlaceholder)) {
              const multilineOutput: string[] = [];
              for (const [_nestedKey, nestedValue] of Object.entries(value as Record<string, string>)) {
                multilineOutput.push(nestedValue);
              }
              linesOut.push(multilineOutput.join('\\N'));
            } else {
              const expectedKey = (lines as string[])[expectedIndex];
              if (key != expectedKey) {
                log.warn(
                  '[TranslatorStructuredObject]',
                  'Unexpected key',
                  'Expected',
                  expectedKey,
                  'Received',
                  key,
                );
              }
              const element = parsedData[key];
              linesOut.push(element);
            }
            expectedIndex++;
          }
          return linesOut;
        }
      }

      const linesOut = getLinesOutput();

      return TranslationOutput.fromCompletion(linesOut, output);
    } catch (error: any) {
      log.error(
        '[TranslatorStructuredObject]',
        `Error ${error?.constructor?.name}`,
        error?.message,
      );
      return this.handleTranslateError(error, lines.length)!;
    }
  }

  /**
   * Builds assistant context messages by pairing source lines with their
   * translations in a JSON object mapping (source -> translation).
   *
   * @param sourceLines - The original source text lines.
   * @param transformLines - The corresponding translated lines.
   * @returns An array containing a single assistant message with the JSON mapping.
   */
  override getContext(
    sourceLines: string[],
    transformLines: string[],
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const output: Record<string, string> = {};

    for (let index = 0; index < sourceLines.length; index++) {
      const source = sourceLines[index];
      const transform = transformLines[index];
      output[source] = transform;
    }

    return [{ role: 'assistant' as const, content: JSON.stringify(output) }];
  }
}
