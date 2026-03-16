import { OpenAI } from 'openai';
import type { LLMProvider } from './types.js';

export interface OpenAIProviderOptions {
  apiKey: string;
  baseURL?: string;
  dangerouslyAllowBrowser?: boolean;
  proxyAgent?: any;
}

export class OpenAIProvider implements LLMProvider {
  readonly name = 'OpenAI';
  readonly defaultModel: string;
  readonly supportsStructuredOutput = true;
  readonly supportsModeration = true;
  readonly supportsStreaming = true;
  readonly supportsPromptCaching = true;
  readonly systemSuffix = '';

  private client: OpenAI;

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

  getClient(): OpenAI {
    return this.client;
  }
}
