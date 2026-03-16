import type { OpenAI } from 'openai';
import log from 'loglevel';
import type { CooldownContext } from './cooldown.js';
import { openaiRetryWrapper } from './openai.js';

/**
 * A single flagged moderation category and its confidence score.
 */
export interface ModerationResult {
  /** The name of the flagged moderation category (e.g. `"hate"`, `"violence"`). */
  catergory: string;
  /** The confidence score for this category (0 to 1). */
  value: number;
}

/**
 * Service dependencies required for moderation checks.
 */
export interface ModerationServiceContext {
  /** The OpenAI client instance used to call the moderations endpoint. */
  openai: OpenAI;
  /** Optional rate-limiter applied before each moderation request. */
  cooler?: CooldownContext;
}

/**
 * Sends text to the OpenAI moderations endpoint and returns the result.
 *
 * Automatically retries on transient errors via {@link openaiRetryWrapper} and
 * respects the optional {@link CooldownContext} rate limiter.
 *
 * @param input - A single string or array of strings to moderate.
 * @param services - Service context containing the OpenAI client and optional cooldown.
 * @param model - Optional moderation model override.
 * @returns The moderation result for the first input, or `undefined` if all retries fail.
 */
export async function checkModeration(
  input: string | string[],
  services: ModerationServiceContext,
  model?: OpenAI.ModerationModel,
) {
  return await openaiRetryWrapper(
    async () => {
      await services.cooler?.cool();
      const moderation = await services.openai.moderations.create({ input, model });
      const moderationData = moderation.results[0];

      if (moderationData.flagged) {
        log.debug('[CheckModeration]', 'flagged', getModeratorResults(moderationData));
      }

      return moderationData;
    },
    3,
    'CheckModeration',
  );
}

/**
 * Extracts the flagged categories and their scores from a moderation response.
 *
 * @param moderatorOutput - A single {@link OpenAI.Moderation} result object.
 * @returns An array of {@link ModerationResult} entries for every flagged category.
 */
export function getModeratorResults(moderatorOutput: OpenAI.Moderation): ModerationResult[] {
  return Object.keys(moderatorOutput.categories)
    .filter((x) => (moderatorOutput.categories as any)[x])
    .map((x) => ({ catergory: x, value: Number((moderatorOutput.category_scores as any)[x]) }));
}

/**
 * Formats an array of moderation results into a human-readable summary string.
 *
 * @param moderatorResults - The flagged moderation results to describe.
 * @returns A space-separated string of `"category: score"` pairs.
 * @example
 * ```ts
 * getModeratorDescription([{ catergory: 'hate', value: 0.95 }]);
 * // "hate: 0.950"
 * ```
 */
export function getModeratorDescription(moderatorResults: ModerationResult[]): string {
  return moderatorResults.map((x) => `${x.catergory}: ${x.value.toFixed(3)}`).join(' ');
}
