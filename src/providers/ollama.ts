import { OpenAI } from 'openai';
import type { LLMProvider } from './types.js';

export interface OllamaProviderOptions {
  baseURL?: string;
  model: string;
  name?: string;
  supportsStructuredOutput?: boolean;
  systemSuffix?: string;
  dangerouslyAllowBrowser?: boolean;
  prefixNumber?: boolean;
  lineMatching?: boolean;
}

export class OllamaProvider implements LLMProvider {
  readonly name: string;
  readonly defaultModel: string;
  readonly supportsStructuredOutput: boolean;
  readonly supportsModeration = false;
  readonly supportsStreaming = true;
  readonly supportsPromptCaching = false;
  readonly systemSuffix: string;
  readonly prefixNumber: boolean;
  readonly lineMatching: boolean;

  private client: OpenAI;

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

  getClient(): OpenAI {
    return this.client;
  }
}

/** Qwen3 32B (Q4_K_M quantization) via Ollama, with thinking disabled */
export class OllamaQwen3_32B extends OllamaProvider {
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

const TRANSLATE_GEMMA_SUFFIX =
  'Output ONLY the translated text. One line per input line. No explanations, no numbering, no original text.';

/** TranslateGemma 12B via Ollama */
export class OllamaTranslateGemma12B extends OllamaProvider {
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

/** TranslateGemma 4B via Ollama */
export class OllamaTranslateGemma4B extends OllamaProvider {
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
