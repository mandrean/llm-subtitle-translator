import type { OpenAI } from 'openai';

export class TranslationOutput<TContent = string[]> {
  content: TContent;
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  totalTokens: number;
  refusal: string;

  constructor(
    content: TContent,
    promptTokens: number | undefined,
    completionTokens: number | undefined,
    cachedTokens: number | undefined,
    totalTokens?: number,
    refusal: string = '',
  ) {
    this.content = content;
    this.promptTokens = promptTokens ?? 0;
    this.completionTokens = completionTokens ?? 0;
    this.cachedTokens = cachedTokens ?? 0;
    this.totalTokens = totalTokens ?? (this.promptTokens + this.completionTokens);
    this.refusal = refusal;
  }

  static fromCompletion<C = string[]>(
    content: C,
    completion: OpenAI.Chat.ChatCompletion,
  ): TranslationOutput<C> {
    const usage = completion.usage;
    return new TranslationOutput(
      content,
      usage?.prompt_tokens,
      usage?.completion_tokens,
      usage?.prompt_tokens_details?.cached_tokens,
      usage?.total_tokens,
      completion.choices[0]?.message?.refusal ?? undefined,
    );
  }

  static fromUsage<C = string[]>(
    content: C,
    usage: OpenAI.Completions.CompletionUsage | undefined,
  ): TranslationOutput<C> {
    return new TranslationOutput(
      content,
      usage?.prompt_tokens,
      usage?.completion_tokens,
      usage?.prompt_tokens_details?.cached_tokens,
      usage?.total_tokens,
    );
  }
}
