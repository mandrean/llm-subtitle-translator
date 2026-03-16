import type { OpenAI } from 'openai';

/**
 * Abstraction for LLM providers (OpenAI, Ollama, etc.).
 * Each provider produces an OpenAI-compatible client that the translator
 * infrastructure can use directly.
 */
export interface LLMProvider {
  readonly name: string;
  readonly defaultModel: string;
  readonly supportsStructuredOutput: boolean;
  readonly supportsModeration: boolean;
  readonly supportsStreaming: boolean;
  readonly supportsPromptCaching: boolean;
  /** Optional suffix appended to system instructions (e.g. '/no_think' for Qwen3) */
  readonly systemSuffix: string;
  /** Whether this provider works well with numbered line prefixes (default: true) */
  readonly prefixNumber: boolean;
  /** Whether to enforce 1:1 line matching between input and output (default: true) */
  readonly lineMatching: boolean;

  /** Returns an OpenAI-compatible client */
  getClient(): OpenAI;
}
