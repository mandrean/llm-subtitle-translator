import type { OpenAI } from 'openai';
import log from 'loglevel';
import { countTokens } from 'gpt-tokenizer';
import { openaiRetryWrapper, completeChatStream } from './openai.js';
import { checkModeration } from './moderator.js';
import { splitStringByNumberLabel } from './subtitle.js';
import { TranslatorBase, DefaultOptions } from './translatorBase.js';
import type { TranslationServiceContext, TranslatorOptions } from './translatorBase.js';
import { TranslationOutput } from './translatorOutput.js';

export { DefaultOptions };
export type { TranslationServiceContext, TranslatorOptions };

interface WorkingEntry {
  source: string;
  transform: string;
  completionTokens?: number;
}

export class Translator<T = string, TLines extends T[] = T[]> extends TranslatorBase<T, TLines> {
  workingProgress: WorkingEntry[];
  offset: number;
  end: number | undefined;
  moderatorFlags: Map<number, any>;
  workingLines: string[];

  thinkTags: { start: string; end: string };

  constructor(
    language: { from?: string; to: string },
    services: TranslationServiceContext,
    options?: Partial<TranslatorOptions>,
  ) {
    super(language, services, options);
    this.workingProgress = [];
    this.offset = 0;
    this.end = undefined;
    this.moderatorFlags = new Map();
    this.workingLines = [];
    this.thinkTags = {
      start: '<think>',
      end: '</think>',
    };
  }

  getOutput(inputLines: any[], rawContent: string): string[] {
    rawContent = rawContent.trim();
    if (rawContent.startsWith(this.thinkTags.start)) {
      const endTagIndex = rawContent.indexOf(this.thinkTags.end);
      if (endTagIndex > 0) {
        const endIndex = endTagIndex + this.thinkTags.end.length;
        const thinkBlock = rawContent.slice(0, endIndex).trim();
        if (thinkBlock) {
          log.debug('[Translator]', '[ThinkBlock] Detected\n', thinkBlock);
        }
        rawContent = rawContent.slice(endIndex);
      }
    }
    if (inputLines.length === 1) {
      return [rawContent.split('\n').join(' ')];
    } else {
      return rawContent.split('\n').filter((x) => x.trim().length > 0);
    }
  }

  async doTranslatePrompt(lines: TLines): Promise<TranslationOutput> {
    const text = (lines as unknown as string[]).join('\n\n');
    const userMessage: OpenAI.Chat.ChatCompletionMessageParam = {
      role: 'user',
      content: `${text}`,
    };
    const systemMessage: OpenAI.Chat.ChatCompletionMessageParam[] = this.systemInstruction
      ? [{ role: 'system', content: `${this.systemInstruction}` }]
      : [];
    const messages = [...systemMessage, ...this.options.initialPrompts, ...this.promptContext, userMessage];
    const max_tokens = this.getMaxToken(lines);

    const streamMode = this.options.createChatCompletionRequest.stream;
    return (await openaiRetryWrapper(
      async () => {
        await this.services.cooler?.cool();
        if (!streamMode) {
          const promptResponse = await this.client.chat.completions.create({
            messages,
            ...this.options.createChatCompletionRequest,
            stream: false,
            max_tokens,
          } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming);
          const rawContent = promptResponse.choices[0].message.content!;
          return TranslationOutput.fromUsage(this.getOutput(lines as any, rawContent), promptResponse.usage);
        } else {
          const promptResponse = await this.client.chat.completions.create({
            messages,
            ...this.options.createChatCompletionRequest,
            stream: true,
            stream_options: { include_usage: true },
            max_tokens,
          } as OpenAI.Chat.ChatCompletionCreateParamsStreaming);

          this.streamController = promptResponse.controller;

          let writeQueue = '';
          let usage: OpenAI.Completions.CompletionUsage | undefined;
          const streamOutput = await completeChatStream(
            promptResponse,
            (data: string) => {
              const hasNewline = data.includes('\n');
              if (writeQueue.length === 0 && !hasNewline) {
                this.services.onStreamChunk?.(data);
              } else if (hasNewline) {
                writeQueue += data;
                writeQueue = writeQueue.replaceAll('\n\n', '\n');
              } else {
                writeQueue += data;
                this.services.onStreamChunk?.(writeQueue);
                writeQueue = '';
              }
            },
            (u) => {
              usage = u;
              this.services.onStreamEnd?.();
            },
          );
          return TranslationOutput.fromUsage(this.getOutput(lines as any, streamOutput), usage);
        }
      },
      3,
      'TranslationPrompt',
    ))!;
  }

  async *translateSingle(batch: string[]) {
    log.debug(`[Translator]`, 'Single line mode');
    batch = batch.slice(-this.currentBatchSize);
    for (let x = 0; x < batch.length; x++) {
      const input = batch[x];
      this.buildContext();
      const output = await this.translatePrompt([input] as unknown as TLines);
      const writeOut = (output.content as unknown as string[])[0];
      yield* this.yieldOutput([batch[x]], [writeOut], output.completionTokens);
    }
  }

  async *translateLines(lines: string[]) {
    log.debug('[Translator]', 'System Instruction:', this.systemInstruction);
    this.aborted = false;
    this.workingLines = lines;
    const theEnd = this.end ?? lines.length;

    for (
      let index = this.offset, reducedBatchSessions = 0;
      index < theEnd;
      index += this.currentBatchSize
    ) {
      let batch = lines
        .slice(index, index + this.currentBatchSize)
        .map((x, i) => this.preprocessLine(x, i, index));

      if (this.options.useModerator && !this.services.moderationService) {
        log.warn(
          '[Translator]',
          'Moderation service requested but not configured, no moderation applied',
        );
      }

      if (this.options.useModerator && this.services.moderationService) {
        const inputForModeration = batch.join('\n\n');
        const moderationData = await checkModeration(
          inputForModeration,
          this.services.moderationService,
          this.options.moderationModel,
        );
        if (moderationData?.flagged) {
          if (!this.changeBatchSize('decrease')) {
            yield* this.translateSingle(batch);
          } else {
            index -= this.currentBatchSize;
          }
          continue;
        }
      }
      this.buildContext();
      const output = await this.translatePrompt(batch as unknown as TLines);

      if (this.aborted) {
        log.debug('[Translator]', 'Aborted');
        return;
      }

      let outputs = output.content as unknown as string[];

      if (
        (this.options.lineMatching && batch.length !== outputs.length) ||
        (batch.length > 1 && output.refusal)
      ) {
        this.promptTokensWasted += output.promptTokens;
        this.completionTokensWasted += output.completionTokens;

        if (output.refusal) {
          log.debug(`[Translator]`, 'Refusal: ', output.refusal);
        } else {
          log.debug(`[Translator]`, 'Lines count mismatch', batch.length, outputs.length);
        }

        log.debug(`[Translator]`, 'batch', batch);
        log.debug(`[Translator]`, 'transformed', outputs);

        if (this.changeBatchSize('decrease')) {
          index -= this.currentBatchSize;
        } else {
          yield* this.translateSingle(batch);
        }
      } else {
        yield* this.yieldOutput(batch, outputs, output.completionTokens / outputs.length);

        if (this.batchSizeThreshold && reducedBatchSessions++ >= this.batchSizeThreshold) {
          reducedBatchSessions = 0;
          const old = this.currentBatchSize;
          this.changeBatchSize('increase');
          index -= this.currentBatchSize - old;
        }
      }

      this.printUsage();
    }
  }

  *yieldOutput(
    promptSources: string[],
    promptTransforms: string[],
    completionTokensPerEntry?: number,
  ) {
    for (let index = 0; index < promptSources.length; index++) {
      const promptSource = promptSources[index];
      const promptTransform = promptTransforms[index] ?? '';
      const workingIndex = this.workingProgress.length;
      const originalSource = this.workingLines[workingIndex];
      let finalTransform = promptTransform;
      let outTransform = promptTransform;

      if (this.moderatorFlags.has(workingIndex)) {
        finalTransform = `[Flagged][Moderator] ${originalSource} -> ${finalTransform} `;
      } else if (this.options.prefixNumber) {
        const splits = this.postprocessNumberPrefixedLine(finalTransform);
        finalTransform = splits.text;
        outTransform = splits.text;
        const expectedLabel = workingIndex + 1;
        if (expectedLabel !== splits.number) {
          log.warn('[Translator]', 'Label mismatch', expectedLabel, splits.number);
          this.moderatorFlags.set(workingIndex, {
            remarks: 'Label Mismatch',
            outIndex: splits.number,
          });
          finalTransform = `[Flagged][Model] ${originalSource} -> ${finalTransform}`;
        }
      } else {
        finalTransform = this.postprocessLine(finalTransform);
      }
      this.workingProgress.push({
        source: promptSource,
        transform: promptTransform,
        completionTokens: completionTokensPerEntry,
      });
      const output = {
        index: this.workingProgress.length,
        source: originalSource,
        transform: outTransform,
        finalTransform,
      };
      yield output;
    }
  }

  preprocessLine(line: string, index: number, offset: number): string {
    line = line.replaceAll('\n', ' \\N ');
    if (this.options.prefixNumber) {
      line = `${offset + index + 1}. ${line}`;
    }
    return line;
  }

  postprocessNumberPrefixedLine(line: string) {
    const splits = splitStringByNumberLabel(line.trim());
    splits.text = this.postprocessLine(splits.text);
    return splits;
  }

  postprocessLine(line: string): string {
    line = line.replaceAll(' \\N ', '\n');
    line = line.replaceAll('\\N', '\n');
    return line;
  }

  buildContext() {
    if (this.workingProgress.length === 0) {
      return;
    }

    const chunkSize = this.options.batchSizes[this.options.batchSizes.length - 1];

    const allChunks: WorkingEntry[][] = [];
    for (let i = 0; i < this.workingProgress.length; i += chunkSize) {
      allChunks.push(this.workingProgress.slice(i, i + chunkSize));
    }

    const { includedChunks, tokenCount } = this.selectContextChunks(allChunks, (chunk) => {
      const messages = this.getContext(
        chunk.map((e) => e.source),
        chunk.map((e) => e.transform),
      );
      return messages.reduce((sum, m) => sum + countTokens(String(m.content ?? '')), 0);
    });

    const sliced = includedChunks.flat();

    if (this.options.useFullContext > 0) {
      const logSliceContext =
        sliced.length < this.workingProgress.length
          ? `sliced ${this.workingProgress.length - sliced.length} entries (${sliced.length}/${this.workingProgress.length} kept, ${tokenCount} tokens)`
          : `all (${sliced.length} entries, ${tokenCount} tokens)`;
      log.debug('[Translator]', 'Context:', logSliceContext);
    }

    const offset = this.workingProgress.length - sliced.length;

    const checkFlaggedMapper = (text: string, index: number): string => {
      const id = index + (offset < 0 ? 0 : offset);
      if (this.moderatorFlags.has(id)) {
        return this.preprocessLine('-', id, 0);
      }
      return text;
    };

    const checkedSource = sliced.map((x, i) => checkFlaggedMapper(x.source, i));
    const checkedTransform = sliced.map((x, i) => checkFlaggedMapper(x.transform, i));
    this.promptContext = this.getContext(checkedSource, checkedTransform);
  }

  getContext(
    sourceLines: string[],
    transformLines: string[],
  ): OpenAI.Chat.ChatCompletionMessageParam[] {
    const chunks: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    const chunkSize = this.options.batchSizes[this.options.batchSizes.length - 1];
    for (let i = 0; i < sourceLines.length; i += chunkSize) {
      const sourceChunk = sourceLines.slice(i, i + chunkSize);
      const transformChunk = transformLines.slice(i, i + chunkSize);
      chunks.push({
        role: 'user',
        content: this.getContextLines(sourceChunk, 'user'),
      });
      chunks.push({
        role: 'assistant',
        content: this.getContextLines(transformChunk, 'assistant'),
      });
    }
    return chunks;
  }

  getContextLines(lines: string[], _role: 'user' | 'assistant'): string {
    return lines.join('\n\n');
  }
}
