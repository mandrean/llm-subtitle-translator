import test from 'node:test';
import assert from 'node:assert';

import { OpenAIProvider } from '../src/main.js';

import 'dotenv/config';
const provider = new OpenAIProvider({ apiKey: process.env.OPENAI_API_KEY ?? '' });
const openai = provider.getClient();

test('should list available openai models', async () => {
  const models = await openai.models.list();
  const gptModels = models.data.filter(
    (x) =>
      x.id.startsWith('gpt') &&
      !x.id.includes('instruct') &&
      !x.id.includes('vision'),
  );

  const gptList = gptModels
    .map((x) => x.id)
    .sort()
    .join(' ');
  console.log('GPT Text Models:', gptList);
  assert(
    gptModels.length > 0,
    `should have available gpt models: ${gptModels.length} ${gptList}`,
  );
});
