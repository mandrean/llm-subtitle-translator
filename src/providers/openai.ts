import { OpenAI } from 'openai';
import type { LLMProvider } from './types.js';

/**
 * Configuration options for {@link OpenAIProvider}.
 *
 * @example
 * ```ts
 * const opts: OpenAIProviderOptions = {
 *   apiKey: process.env.OPENAI_API_KEY!,
 *   baseURL: 'https://api.openai.com/v1',
 * };
 * ```
 */
export interface OpenAIProviderOptions {
  /** OpenAI API key used for authentication. */
  apiKey: string;
  /** Custom API base URL. Useful for proxies or Azure OpenAI endpoints. */
  baseURL?: string;
  /** When `true`, allows API calls from a browser context. Defaults to `false`. */
  dangerouslyAllowBrowser?: boolean;
  /** Optional HTTP proxy agent (e.g. `undici.ProxyAgent`) forwarded as a fetch dispatcher. */
  proxyAgent?: any;
}

/**
 * LLM provider backed by the OpenAI API.
 *
 * Supports structured output, moderation, streaming, and prompt caching.
 * Uses numbered line prefixes and 1:1 line matching by default.
 *
 * @example
 * ```ts
 * const provider = new OpenAIProvider({ apiKey: 'sk-...' });
 * const client = provider.getClient();
 * ```
 */
export class OpenAIProvider implements LLMProvider {
  /** @inheritdoc */
  readonly name = 'OpenAI';
  /** @inheritdoc */
  readonly defaultModel: string;
  /** @inheritdoc */
  readonly supportsStructuredOutput = true;
  /** @inheritdoc */
  readonly supportsModeration = true;
  /** @inheritdoc */
  readonly supportsStreaming = true;
  /** @inheritdoc */
  readonly supportsPromptCaching = true;
  /** @inheritdoc */
  readonly systemSuffix = '';
  /** @inheritdoc */
  readonly prefixNumber = true;
  /** @inheritdoc */
  readonly lineMatching = true;

  /** The underlying OpenAI client instance. */
  private client: OpenAI;

  /**
   * Creates a new OpenAI provider.
   *
   * @param options - Connection and authentication settings.
   * @param defaultModel - Model identifier to use when none is specified. Defaults to `'gpt-4o-mini'`.
   */
  constructor(options: OpenAIProviderOptions, defaultModel?: string) {
    this.defaultModel = defaultModel ?? 'gpt-4o-mini';
    this.client = new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      dangerouslyAllowBrowser: options.dangerouslyAllowBrowser,
      maxRetries: 3,
      fetchOptions: options.proxyAgent ? { dispatcher: options.proxyAgent as any } : undefined,
    });
  }

  /**
   * Returns the underlying OpenAI client.
   *
   * @returns The configured {@link OpenAI} client instance.
   */
  getClient(): OpenAI {
    return this.client;
  }
}
