import type { OpenAI } from 'openai';
import log from 'loglevel';
import type { CooldownContext } from './cooldown.js';
import { openaiRetryWrapper } from './openai.js';

export interface ModerationResult {
  catergory: string;
  value: number;
}

export interface ModerationServiceContext {
  openai: OpenAI;
  cooler?: CooldownContext;
}

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

export function getModeratorResults(moderatorOutput: OpenAI.Moderation): ModerationResult[] {
  return Object.keys(moderatorOutput.categories)
    .filter((x) => (moderatorOutput.categories as any)[x])
    .map((x) => ({ catergory: x, value: Number((moderatorOutput.category_scores as any)[x]) }));
}

export function getModeratorDescription(moderatorResults: ModerationResult[]): string {
  return moderatorResults.map((x) => `${x.catergory}: ${x.value.toFixed(3)}`).join(' ');
}
