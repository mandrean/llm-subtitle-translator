import type { OpenAI } from 'openai';
import log from 'loglevel';
import { countTokens } from 'gpt-tokenizer';
import { roundWithPrecision, sleep } from './helpers.js';
import type { LLMProvider } from './providers/types.js';
import type { CooldownContext } from './cooldown.js';
import type { ModerationServiceContext } from './moderator.js';
import type { TranslationOutput } from './translatorOutput.js';

/**
 * Context object providing external services to the translator,
 * including the LLM provider, rate-limiting, streaming callbacks, and moderation.
 */
export interface TranslationServiceContext {
  /** The LLM provider used to obtain an OpenAI-compatible client. */
  provider: LLMProvider;
  /** Optional cooldown/rate-limiter to throttle API requests. */
  cooler?: CooldownContext;
  /** Callback invoked with each chunk of streamed output text. */
  onStreamChunk?: (data: string) => void;
  /** Callback invoked when a streaming response finishes. */
  onStreamEnd?: () => void;
  /** Callback invoked to clear the current terminal line during streaming. */
  onClearLine?: () => void;
  /** Optional moderation service for content-policy checks. */
  moderationService?: ModerationServiceContext;
}

/**
 * Configuration options controlling translator behaviour, including model
 * parameters, batching strategy, token limits, and output format.
 */
export interface TranslatorOptions {
  /** Partial chat-completion request parameters merged with defaults before each API call. */
  createChatCompletionRequest: Partial<OpenAI.Chat.ChatCompletionCreateParams> & { model?: string };
  /** Model identifier used for the OpenAI moderation endpoint. */
  moderationModel: OpenAI.ModerationModel;
  /** Few-shot or instruction messages prepended to every prompt. */
  initialPrompts: OpenAI.Chat.ChatCompletionMessageParam[];
  /** Whether to run content through the moderation endpoint before translating. */
  useModerator: boolean;
  /** Whether to prefix each line with a sequential number for alignment verification. */
  prefixNumber: boolean;
  /** Whether to enforce that the number of output lines matches the input. */
  lineMatching: boolean;
  /** Maximum token budget for context (prior translations). 0 disables context limiting. */
  useFullContext: number;
  /** Ordered list of batch sizes; the translator cycles through them on mismatch. */
  batchSizes: number[];
  /** Structured output format mode, or `false`/`'none'` for plain text. */
  structuredMode: 'array' | 'object' | 'none' | 'timestamp' | 'agent' | false;
  /** Hard cap on `max_tokens` sent to the model. 0 means no cap. */
  max_token: number;
  /** Multiplier applied to input token count to derive a dynamic `max_tokens` value. */
  inputMultiplier: number;
  /** Log verbosity level passed to the `loglevel` library. */
  logLevel: log.LogLevelDesc | undefined;
  /** Optional path to the input subtitle file. */
  inputFile?: string;
}

/** Sensible default values for all translator options. */
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

/**
 * Abstract base class for subtitle translators.
 *
 * Manages token accounting, batch-size adaptation, context window selection,
 * and abort signalling. Concrete subclasses implement the actual prompt
 * construction and line-level translation logic.
 *
 * @typeParam T      - The element type of a single translation unit (default `string`).
 * @typeParam TLines - The array type passed to translation methods (default `T[]`).
 */
export abstract class TranslatorBase<T = string, TLines extends T[] = T[]> {
  /** Source and target language descriptors. */
  language: { from?: string; to: string };
  /** External services (provider, cooldown, moderation, streaming callbacks). */
  services: TranslationServiceContext;
  /** Merged translator options with a guaranteed `model` field. */
  options: TranslatorOptions & { createChatCompletionRequest: { model: string } };
  /** System-level instruction prepended to every chat prompt. */
  systemInstruction: string;
  /** Accumulated context messages (prior source/translation pairs) for few-shot prompting. */
  promptContext: OpenAI.Chat.ChatCompletionMessageParam[];

  /** Total prompt tokens consumed by successful translations. */
  promptTokensUsed: number;
  /** Prompt tokens consumed by failed/retried translations. */
  promptTokensWasted: number;
  /** Tokens served from the provider's prompt cache. */
  cachedTokens: number;
  /** Total completion tokens consumed by successful translations. */
  completionTokensUsed: number;
  /** Completion tokens consumed by failed/retried translations. */
  completionTokensWasted: number;
  /** Cumulative wall-clock time (ms) spent waiting for model responses. */
  tokensProcessTimeMs: number;
  /** Prompt tokens used in the most recent translation call. */
  contextPromptTokens: number;
  /** Completion tokens used in the most recent translation call. */
  contextCompletionTokens: number;

  /** Rotating queue of batch sizes used for adaptive batching. */
  workingBatchSizes: number[];
  /** The batch size currently in use. */
  currentBatchSize: number;
  /** Number of successful batches required before attempting to increase the batch size. */
  batchSizeThreshold: number | undefined;

  /** Whether the translation run has been aborted. */
  aborted: boolean;
  /** Controller for aborting an in-flight streaming request. */
  streamController: AbortController | undefined;

  /** OpenAI-compatible client obtained from the provider */
  protected client: OpenAI;

  /**
   * Creates a new translator instance.
   *
   * Merges caller-supplied options with {@link DefaultOptions} and any
   * provider-level defaults, initialises token counters and batch-size state,
   * and obtains an OpenAI-compatible client from the provider.
   *
   * @param language - Source (optional) and target language.
   * @param services - External service dependencies.
   * @param options  - Partial overrides for the default translator options.
   */
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

    // Apply provider defaults for options not explicitly set by the caller
    const providerDefaults: Partial<TranslatorOptions> = {};
    if (opts.prefixNumber === undefined && services.provider.prefixNumber !== undefined) {
      providerDefaults.prefixNumber = services.provider.prefixNumber;
    }
    if (opts.lineMatching === undefined && services.provider.lineMatching !== undefined) {
      providerDefaults.lineMatching = services.provider.lineMatching;
    }

    this.language = language;
    this.services = services;
    this.options = { ...DefaultOptions, ...providerDefaults, ...opts } as TranslatorOptions & {
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

  /**
   * Translates an array of subtitle lines, yielding results one at a time.
   *
   * @param _lines - Raw subtitle text lines to translate.
   * @returns An async generator yielding per-line translation results.
   */
  abstract translateLines(
    _lines: string[],
  ): AsyncGenerator<{ index: number; source: string; transform: string; finalTransform: string }>;

  /**
   * Translates a batch of lines and records token usage and elapsed time.
   *
   * Delegates to {@link doTranslatePrompt} and wraps the call with timing
   * so that {@link accumulateUsage} can update cumulative counters.
   *
   * @param lines - The batch of lines to translate.
   * @returns The translation output including content and token counts.
   */
  async translatePrompt(lines: TLines): Promise<TranslationOutput> {
    const startTime = Date.now();
    const output = await this.doTranslatePrompt(lines);
    const endTime = Date.now();
    const result = this.accumulateUsage(output, endTime - startTime);
    return result;
  }

  /**
   * Performs the actual model call for a batch of lines.
   *
   * Subclasses must implement this to construct the prompt, call the LLM,
   * and return a {@link TranslationOutput}.
   *
   * @param _lines - The batch of lines to translate.
   * @returns The raw translation output from the model.
   */
  abstract doTranslatePrompt(_lines: TLines): Promise<TranslationOutput>;

  /**
   * Computes the `max_tokens` value to pass to the model for a given batch.
   *
   * When both {@link TranslatorOptions.max_token} and
   * {@link TranslatorOptions.inputMultiplier} are set, the effective limit is
   * the lesser of the hard cap and the input token count multiplied by the
   * multiplier. Returns `undefined` when no limit applies.
   *
   * @param lines - The batch of lines whose token count is measured.
   * @returns The computed max-token limit, or `undefined` if uncapped.
   */
  getMaxToken(lines: TLines): number | undefined {
    if (this.options.max_token && !this.options.inputMultiplier) {
      return this.options.max_token;
    } else if (this.options.max_token && this.options.inputMultiplier) {
      const max = countTokens(JSON.stringify(lines)) * this.options.inputMultiplier;
      return Math.min(this.options.max_token, max);
    }
    return undefined;
  }

  /**
   * Adjusts the current batch size up or down by rotating the working batch-size queue.
   *
   * When decreasing, the largest size is moved to the front; when increasing,
   * the smallest is moved to the back. Also updates
   * {@link batchSizeThreshold} so the translator knows how many successful
   * batches to wait before trying a larger size again.
   *
   * @param mode - `'increase'` to try a larger batch, `'decrease'` to shrink.
   * @returns `true` if the batch size changed, `false` if already at the limit.
   */
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

  /**
   * Adds the token counts from a single translation call to the running totals.
   *
   * Also stores the per-call counts in {@link contextPromptTokens} and
   * {@link contextCompletionTokens} for the next usage report.
   *
   * @param output    - The translation output whose usage to accumulate.
   * @param elapsedMs - Wall-clock milliseconds the call took.
   * @returns The same `output` object, for chaining convenience.
   */
  accumulateUsage(output: TranslationOutput, elapsedMs: number): TranslationOutput {
    this.promptTokensUsed += output.promptTokens;
    this.completionTokensUsed += output.completionTokens;
    this.cachedTokens += output.cachedTokens;
    this.contextPromptTokens = output.promptTokens;
    this.contextCompletionTokens = output.completionTokens;
    this.tokensProcessTimeMs += elapsedMs;
    return output;
  }

  /**
   * Selects as many trailing context chunks as fit within the token budget.
   *
   * Iterates backward through `chunks`, summing token costs until
   * {@link TranslatorOptions.useFullContext} would be exceeded. When the
   * budget is zero or negative, exactly one chunk is included as a minimum.
   *
   * @typeParam C - The chunk type.
   * @param chunks       - Ordered array of context chunks (oldest first).
   * @param getChunkCost - Function returning the token cost of a single chunk.
   * @returns The selected chunks and the total token count they consume.
   */
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

  /**
   * Computes a snapshot of cumulative token usage statistics.
   *
   * Includes total and wasted tokens, cached tokens, per-call context tokens,
   * tokens-per-minute rates, and a human-readable wasted-percentage string.
   */
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

  /**
   * Logs a formatted summary of token usage to the debug output.
   *
   * Includes used/wasted/cached token counts, context utilisation percentage,
   * and tokens-per-minute throughput alongside the cooldown rate.
   */
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

  /**
   * Signals the translator to stop after the current in-flight request.
   *
   * Aborts any active streaming response via its {@link streamController}
   * and sets the {@link aborted} flag so the translation loop exits.
   */
  abort() {
    log.warn('[Translator]', 'Aborting');
    this.streamController?.abort();
    this.aborted = true;
  }
}
