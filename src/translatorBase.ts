import type { OpenAI } from 'openai';
import log from 'loglevel';
import { countTokens } from 'gpt-tokenizer';
import { roundWithPrecision, sleep } from './helpers.js';
import type { LLMProvider } from './providers/types.js';
import type { CooldownContext } from './cooldown.js';
import type { ModerationServiceContext } from './moderator.js';
import type { TranslationOutput } from './translatorOutput.js';

export interface TranslationServiceContext {
  provider: LLMProvider;
  cooler?: CooldownContext;
  onStreamChunk?: (data: string) => void;
  onStreamEnd?: () => void;
  onClearLine?: () => void;
  moderationService?: ModerationServiceContext;
}

export interface TranslatorOptions {
  createChatCompletionRequest: Partial<OpenAI.Chat.ChatCompletionCreateParams> & { model?: string };
  moderationModel: OpenAI.ModerationModel;
  initialPrompts: OpenAI.Chat.ChatCompletionMessageParam[];
  useModerator: boolean;
  prefixNumber: boolean;
  lineMatching: boolean;
  useFullContext: number;
  batchSizes: number[];
  structuredMode: 'array' | 'object' | 'none' | 'timestamp' | 'agent' | false;
  max_token: number;
  inputMultiplier: number;
  logLevel: log.LogLevelDesc | undefined;
  inputFile?: string;
}

export const DefaultOptions: TranslatorOptions = {
  createChatCompletionRequest: {
    model: 'gpt-4o-mini',
    temperature: 0,
  },
  moderationModel: 'omni-moderation-latest',
  initialPrompts: [],
  useModerator: false,
  prefixNumber: true,
  lineMatching: true,
  useFullContext: 2000,
  batchSizes: [10, 50],
  structuredMode: 'array',
  max_token: 0,
  inputMultiplier: 0,
  logLevel: undefined,
};

export abstract class TranslatorBase<T = string, TLines extends T[] = T[]> {
  language: { from?: string; to: string };
  services: TranslationServiceContext;
  options: TranslatorOptions & { createChatCompletionRequest: { model: string } };
  systemInstruction: string;
  promptContext: OpenAI.Chat.ChatCompletionMessageParam[];

  promptTokensUsed: number;
  promptTokensWasted: number;
  cachedTokens: number;
  completionTokensUsed: number;
  completionTokensWasted: number;
  tokensProcessTimeMs: number;
  contextPromptTokens: number;
  contextCompletionTokens: number;

  workingBatchSizes: number[];
  currentBatchSize: number;
  batchSizeThreshold: number | undefined;

  aborted: boolean;
  streamController: AbortController | undefined;

  /** OpenAI-compatible client obtained from the provider */
  protected client: OpenAI;

  constructor(
    language: { from?: string; to: string },
    services: TranslationServiceContext,
    options?: Partial<TranslatorOptions>,
  ) {
    const opts = options ?? {};
    opts.createChatCompletionRequest = {
      ...DefaultOptions.createChatCompletionRequest,
      ...opts.createChatCompletionRequest,
    };

    this.language = language;
    this.services = services;
    this.options = { ...DefaultOptions, ...opts } as TranslatorOptions & {
      createChatCompletionRequest: { model: string };
    };
    this.systemInstruction = `Translate ${this.language.from ? this.language.from + ' ' : ''}to ${this.language.to}`;

    // Apply provider's system suffix (e.g. /no_think for Qwen3)
    if (services.provider.systemSuffix) {
      this.systemInstruction += '\n' + services.provider.systemSuffix;
    }

    this.promptContext = [];
    this.client = services.provider.getClient();

    this.promptTokensUsed = 0;
    this.promptTokensWasted = 0;
    this.cachedTokens = 0;
    this.completionTokensUsed = 0;
    this.completionTokensWasted = 0;
    this.tokensProcessTimeMs = 0;
    this.contextPromptTokens = 0;
    this.contextCompletionTokens = 0;

    this.workingBatchSizes = [...this.options.batchSizes];
    this.currentBatchSize = this.workingBatchSizes[this.workingBatchSizes.length - 1];
    this.batchSizeThreshold = undefined;

    this.aborted = false;
    this.streamController = undefined;

    if (opts.logLevel) {
      log.setLevel(opts.logLevel);
    }
    log.debug('[Translator]', 'Model:', this.options.createChatCompletionRequest.model);
  }

  abstract translateLines(
    _lines: string[],
  ): AsyncGenerator<{ index: number; source: string; transform: string; finalTransform: string }>;

  async translatePrompt(lines: TLines): Promise<TranslationOutput> {
    const startTime = Date.now();
    const output = await this.doTranslatePrompt(lines);
    const endTime = Date.now();
    const result = this.accumulateUsage(output, endTime - startTime);
    return result;
  }

  abstract doTranslatePrompt(_lines: TLines): Promise<TranslationOutput>;

  getMaxToken(lines: TLines): number | undefined {
    if (this.options.max_token && !this.options.inputMultiplier) {
      return this.options.max_token;
    } else if (this.options.max_token && this.options.inputMultiplier) {
      const max = countTokens(JSON.stringify(lines)) * this.options.inputMultiplier;
      return Math.min(this.options.max_token, max);
    }
    return undefined;
  }

  changeBatchSize(mode: 'increase' | 'decrease'): boolean {
    const old = this.currentBatchSize;
    if (mode === 'decrease') {
      if (this.currentBatchSize === this.options.batchSizes[0]) {
        return false;
      }
      this.workingBatchSizes.unshift(this.workingBatchSizes.pop()!);
    } else if (mode === 'increase') {
      if (this.currentBatchSize === this.options.batchSizes[this.options.batchSizes.length - 1]) {
        return false;
      }
      this.workingBatchSizes.push(this.workingBatchSizes.shift()!);
    }
    this.currentBatchSize = this.workingBatchSizes[this.workingBatchSizes.length - 1];
    if (this.currentBatchSize === this.options.batchSizes[this.options.batchSizes.length - 1]) {
      this.batchSizeThreshold = undefined;
    } else {
      this.batchSizeThreshold = Math.floor(
        Math.max(old, this.currentBatchSize) / Math.min(old, this.currentBatchSize),
      );
    }
    log.debug(
      '[Translator]',
      'BatchSize',
      mode,
      old,
      '->',
      this.currentBatchSize,
      'SizeThreshold',
      this.batchSizeThreshold,
    );
    return true;
  }

  accumulateUsage(output: TranslationOutput, elapsedMs: number): TranslationOutput {
    this.promptTokensUsed += output.promptTokens;
    this.completionTokensUsed += output.completionTokens;
    this.cachedTokens += output.cachedTokens;
    this.contextPromptTokens = output.promptTokens;
    this.contextCompletionTokens = output.completionTokens;
    this.tokensProcessTimeMs += elapsedMs;
    return output;
  }

  selectContextChunks<C>(
    chunks: C[],
    getChunkCost: (chunk: C) => number,
  ): { includedChunks: C[]; tokenCount: number } {
    const maxTokens = this.options.useFullContext;
    let tokenCount = 0;
    let includedCount = maxTokens <= 0 ? Math.min(1, chunks.length) : 0;
    if (maxTokens > 0) {
      for (let i = chunks.length - 1; i >= 0; i--) {
        const cost = getChunkCost(chunks[i]);
        if (tokenCount + cost > maxTokens) break;
        tokenCount += cost;
        includedCount++;
      }
      if (includedCount === 0 && chunks.length > 0) includedCount = 1;
    }
    return { includedChunks: chunks.slice(chunks.length - includedCount), tokenCount };
  }

  get usage() {
    const promptTokensUsed = this.promptTokensUsed;
    const completionTokensUsed = this.completionTokensUsed;
    const promptTokensWasted = this.promptTokensWasted;
    const completionTokensWasted = this.completionTokensWasted;
    const usedTokens = promptTokensUsed + completionTokensUsed;
    const wastedTokens = promptTokensWasted + completionTokensWasted;
    const minutesElapsed = this.tokensProcessTimeMs / 1000 / 60;
    const promptRate = roundWithPrecision(promptTokensUsed / minutesElapsed, 0);
    const completionRate = roundWithPrecision(completionTokensUsed / minutesElapsed, 0);
    const rate = roundWithPrecision(usedTokens / minutesElapsed, 0);
    const wastedPercent = (wastedTokens / usedTokens).toLocaleString(undefined, {
      style: 'percent',
      minimumFractionDigits: 0,
    });
    const cachedTokens = this.cachedTokens;
    const contextPromptTokens = this.contextPromptTokens;
    const contextCompletionTokens = this.contextCompletionTokens;
    const contextTokens = contextPromptTokens + contextCompletionTokens;
    return {
      promptTokensUsed,
      completionTokensUsed,
      promptTokensWasted,
      completionTokensWasted,
      usedTokens,
      wastedTokens,
      wastedPercent,
      cachedTokens,
      contextPromptTokens,
      contextCompletionTokens,
      contextTokens,
      promptRate,
      completionRate,
      rate,
    };
  }

  async printUsage() {
    const usage = this.usage;

    await sleep(10);

    const {
      promptTokensUsed,
      completionTokensUsed,
      promptTokensWasted,
      completionTokensWasted,
      usedTokens,
      wastedTokens,
      wastedPercent,
      cachedTokens,
      contextPromptTokens,
      contextCompletionTokens,
      contextTokens,
      promptRate,
      completionRate,
      rate,
    } = usage;

    log.debug(
      `[Translator] Estimated Usage`,
      '\n\tTokens:',
      promptTokensUsed,
      '+',
      completionTokensUsed,
      '=',
      usedTokens,
      '\n\tWasted:',
      promptTokensWasted,
      '+',
      completionTokensWasted,
      '=',
      wastedTokens,
      wastedPercent,
      '\n\tCached:',
      cachedTokens >= 0 ? cachedTokens : '-',
      '\n\tContext:',
      ...(contextTokens > 0
        ? [
            contextPromptTokens,
            '+',
            contextCompletionTokens,
            '=',
            contextTokens,
            '/',
            this.options.useFullContext,
            `(${Math.round((contextTokens / this.options.useFullContext) * 100)}%)`,
          ]
        : ['-']),
      '\n\tRate:',
      promptRate,
      '+',
      completionRate,
      '=',
      rate,
      'TPM',
      this.services.cooler?.rate,
      'RPM',
    );
  }

  abort() {
    log.warn('[Translator]', 'Aborting');
    this.streamController?.abort();
    this.aborted = true;
  }
}
