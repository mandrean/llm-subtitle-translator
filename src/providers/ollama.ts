import { OpenAI } from 'openai';
import type { LLMProvider } from './types.js';

/**
 * Configuration options for {@link OllamaProvider}.
 *
 * @example
 * ```ts
 * const opts: OllamaProviderOptions = {
 *   model: 'qwen3:32b',
 *   baseURL: 'http://localhost:11434/v1',
 * };
 * ```
 */
export interface OllamaProviderOptions {
  /** Ollama API base URL. Defaults to `'http://localhost:11434/v1'`. */
  baseURL?: string;
  /** Ollama model tag to use (e.g. `'qwen3:32b'`, `'translategemma:12b'`). */
  model: string;
  /** Human-readable provider name shown in logs. Defaults to `'Ollama (<model>)'`. */
  name?: string;
  /** Whether the model supports structured (JSON schema) output. Defaults to `false`. */
  supportsStructuredOutput?: boolean;
  /** Suffix appended to system instructions (e.g. `'/no_think'`). Defaults to `''`. */
  systemSuffix?: string;
  /** When `true`, allows API calls from a browser context. Defaults to `false`. */
  dangerouslyAllowBrowser?: boolean;
  /** Whether to prepend line numbers to each input line. Defaults to `true`. */
  prefixNumber?: boolean;
  /** Whether to enforce 1:1 line matching between input and output. Defaults to `true`. */
  lineMatching?: boolean;
}

/**
 * LLM provider backed by a local Ollama instance.
 *
 * Communicates with Ollama via its OpenAI-compatible API endpoint.
 * Does not support moderation or prompt caching. Streaming is supported.
 *
 * @example
 * ```ts
 * const provider = new OllamaProvider({ model: 'qwen3:32b' });
 * const client = provider.getClient();
 * ```
 */
export class OllamaProvider implements LLMProvider {
  /** @inheritdoc */
  readonly name: string;
  /** @inheritdoc */
  readonly defaultModel: string;
  /** @inheritdoc */
  readonly supportsStructuredOutput: boolean;
  /** @inheritdoc */
  readonly supportsModeration = false;
  /** @inheritdoc */
  readonly supportsStreaming = true;
  /** @inheritdoc */
  readonly supportsPromptCaching = false;
  /** @inheritdoc */
  readonly systemSuffix: string;
  /** @inheritdoc */
  readonly prefixNumber: boolean;
  /** @inheritdoc */
  readonly lineMatching: boolean;

  /** The underlying OpenAI-compatible client instance. */
  private client: OpenAI;

  /**
   * Creates a new Ollama provider.
   *
   * @param options - Model selection and connection settings. See {@link OllamaProviderOptions}.
   */
  constructor(options: OllamaProviderOptions) {
    const baseURL = options.baseURL ?? 'http://localhost:11434/v1';
    this.name = options.name ?? `Ollama (${options.model})`;
    this.defaultModel = options.model;
    this.supportsStructuredOutput = options.supportsStructuredOutput ?? false;
    this.systemSuffix = options.systemSuffix ?? '';
    this.prefixNumber = options.prefixNumber ?? true;
    this.lineMatching = options.lineMatching ?? true;
    this.client = new OpenAI({
      apiKey: 'ollama',
      baseURL,
      dangerouslyAllowBrowser: options.dangerouslyAllowBrowser,
    });
  }

  /**
   * Returns the underlying OpenAI-compatible client.
   *
   * @returns The configured {@link OpenAI} client instance pointing at the Ollama endpoint.
   */
  getClient(): OpenAI {
    return this.client;
  }
}

/**
 * Pre-configured Ollama provider for the Qwen3 32B model (Q4_K_M quantization).
 *
 * Enables structured output and appends `/no_think` to the system prompt
 * to disable the model's internal chain-of-thought reasoning.
 *
 * @example
 * ```ts
 * const provider = new OllamaQwen3_32B();
 * ```
 */
export class OllamaQwen3_32B extends OllamaProvider {
  /**
   * @param baseURL - Ollama API base URL. Defaults to `'http://localhost:11434/v1'`.
   * @param dangerouslyAllowBrowser - When `true`, allows API calls from a browser context.
   */
  constructor(baseURL?: string, dangerouslyAllowBrowser?: boolean) {
    super({
      model: 'qwen3:32b',
      name: 'Ollama Qwen3 32B (Q4_K_M)',
      baseURL,
      supportsStructuredOutput: true,
      systemSuffix: '/no_think',
      dangerouslyAllowBrowser,
    });
  }
}

/**
 * Suffix appended to system instructions for TranslateGemma models.
 * Constrains output to translated text only, without explanations or numbering.
 */
const TRANSLATE_GEMMA_SUFFIX =
  'Output ONLY the translated text. One line per input line. No explanations, no numbering, no original text.';

/**
 * Pre-configured Ollama provider for the TranslateGemma 12B model.
 *
 * Disables structured output, line numbering, and line matching since
 * TranslateGemma produces plain translated text without numbered prefixes.
 *
 * @example
 * ```ts
 * const provider = new OllamaTranslateGemma12B();
 * ```
 */
export class OllamaTranslateGemma12B extends OllamaProvider {
  /**
   * @param baseURL - Ollama API base URL. Defaults to `'http://localhost:11434/v1'`.
   * @param dangerouslyAllowBrowser - When `true`, allows API calls from a browser context.
   */
  constructor(baseURL?: string, dangerouslyAllowBrowser?: boolean) {
    super({
      model: 'translategemma:12b',
      name: 'Ollama TranslateGemma 12B',
      baseURL,
      supportsStructuredOutput: false,
      systemSuffix: TRANSLATE_GEMMA_SUFFIX,
      prefixNumber: false,
      lineMatching: false,
      dangerouslyAllowBrowser,
    });
  }
}

/**
 * Pre-configured Ollama provider for the TranslateGemma 4B model.
 *
 * Smaller variant of {@link OllamaTranslateGemma12B} with the same configuration.
 * Suitable for machines with limited VRAM.
 *
 * @example
 * ```ts
 * const provider = new OllamaTranslateGemma4B();
 * ```
 */
export class OllamaTranslateGemma4B extends OllamaProvider {
  /**
   * @param baseURL - Ollama API base URL. Defaults to `'http://localhost:11434/v1'`.
   * @param dangerouslyAllowBrowser - When `true`, allows API calls from a browser context.
   */
  constructor(baseURL?: string, dangerouslyAllowBrowser?: boolean) {
    super({
      model: 'translategemma:4b',
      name: 'Ollama TranslateGemma 4B',
      baseURL,
      supportsStructuredOutput: false,
      systemSuffix: TRANSLATE_GEMMA_SUFFIX,
      prefixNumber: false,
      lineMatching: false,
      dangerouslyAllowBrowser,
    });
  }
}
