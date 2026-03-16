/**
 * Integration tests for Ollama providers (translategemma:12b).
 *
 * These tests hit a real local Ollama instance and are SKIPPED by default.
 * To run them manually:
 *
 *   1. Make sure Ollama is running:  ollama serve
 *   2. Pull the model:               ollama pull translategemma:12b
 *   3. Run the tests:                npm run test:ollama
 *
 * Or set OLLAMA_ENABLED=1 to enable them in any test run:
 *   OLLAMA_ENABLED=1 npm run test:all
 */
import fs from 'node:fs';
import { describe, test, before } from 'node:test';
import type { TestContext } from 'node:test';
import assert from 'node:assert';

import {
  subtitleParser,
  Translator,
  CooldownContext,
  OllamaTranslateGemma12B,
} from '../src/main.js';
import { wrapQuotes } from '../src/helpers.js';

import 'dotenv/config';

const OLLAMA_ENABLED = process.env.OLLAMA_ENABLED === '1';
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL;

const provider = new OllamaTranslateGemma12B(OLLAMA_BASE_URL);
const model = provider.defaultModel;
const cooler = new CooldownContext(10, 60000, 'Ollama');

/** Check that the Ollama server is reachable and the model is available. */
async function ollamaHealthCheck(): Promise<boolean> {
  try {
    const client = provider.getClient();
    const models = await client.models.list();
    const available: string[] = [];
    for await (const m of models) {
      available.push(m.id);
    }
    return available.some((id) => id.startsWith('translategemma'));
  } catch {
    return false;
  }
}

function skipUnlessReady(t: TestContext, ready: boolean) {
  if (!ready) {
    t.skip('Ollama not reachable or translategemma:12b not pulled');
  }
}

describe('Ollama integration (translategemma:12b)', { skip: !OLLAMA_ENABLED }, () => {
  let ollamaReady = false;

  before(async () => {
    ollamaReady = await ollamaHealthCheck();
    if (!ollamaReady) {
      console.warn(
        '\n⚠ Ollama is not reachable or translategemma:12b is not pulled — skipping integration tests.\n' +
          '  Run: ollama serve && ollama pull translategemma:12b\n',
      );
    }
  });

  test('plain text: translate a short Chinese text to English', async (t) => {
    skipUnlessReady(t, ollamaReady);

    const translator = new Translator(
      { from: 'Chinese', to: 'English' },
      { cooler, provider },
      {
        createChatCompletionRequest: { model, temperature: 0, stream: false },
        batchSizes: [10],
        prefixNumber: false,
        lineMatching: false,
      },
    );

    const lines = ['你好。', '拜拜！'];
    const outputs: string[] = [];

    for await (const output of translator.translateLines(lines)) {
      console.log(output.index, wrapQuotes(output.source), '->', wrapQuotes(output.finalTransform));
      outputs.push(output.finalTransform);
    }

    assert.strictEqual(outputs.length, lines.length, 'Should produce one output per input line');
    for (const out of outputs) {
      assert.ok(out.trim().length > 0, `Expected non-empty translation, got: "${out}"`);
    }
  });

  test('plain text: translate with streaming enabled', async (t) => {
    skipUnlessReady(t, ollamaReady);

    const chunks: string[] = [];
    const translator = new Translator(
      { from: 'Chinese', to: 'English' },
      {
        cooler,
        provider,
        onStreamChunk: (data) => chunks.push(data),
        onStreamEnd: () => {},
      },
      {
        createChatCompletionRequest: { model, temperature: 0, stream: true },
        batchSizes: [10],
        prefixNumber: false,
        lineMatching: false,
      },
    );

    const lines = ['你好。'];
    const outputs: string[] = [];

    for await (const output of translator.translateLines(lines)) {
      outputs.push(output.finalTransform);
    }

    assert.strictEqual(outputs.length, 1);
    assert.ok(chunks.length > 0, 'Should have received stream chunks');
    assert.ok(outputs[0].trim().length > 0, `Expected non-empty translation, got: "${outputs[0]}"`);
  });

  test('SRT file: translate Japanese subtitles to English', async (t) => {
    skipUnlessReady(t, ollamaReady);

    const fileContent = fs.readFileSync('./test/data/test_ja_small.srt', 'utf-8');
    const srtParsed = subtitleParser.fromSrt(fileContent).map((x) => x.text);

    const translator = new Translator(
      { from: 'Japanese', to: 'English' },
      { cooler, provider },
      {
        createChatCompletionRequest: { model, temperature: 0, stream: false },
        batchSizes: [2, 5],
        prefixNumber: false,
        lineMatching: false,
      },
    );

    const outputs: Array<{ index: number; source: string; finalTransform: string }> = [];

    for await (const output of translator.translateLines(srtParsed)) {
      console.log(output.index, wrapQuotes(output.source), '->', wrapQuotes(output.finalTransform));
      outputs.push(output);
    }

    assert.strictEqual(
      outputs.length,
      srtParsed.length,
      `Expected ${srtParsed.length} outputs, got ${outputs.length}`,
    );

    for (const o of outputs) {
      assert.ok(o.finalTransform.trim().length > 0, `Empty translation for line ${o.index}`);
    }
  });

  test('provider properties are correct', (t) => {
    skipUnlessReady(t, ollamaReady);

    assert.strictEqual(provider.name, 'Ollama TranslateGemma 12B');
    assert.strictEqual(provider.defaultModel, 'translategemma:12b');
    assert.strictEqual(provider.supportsStructuredOutput, false);
    assert.strictEqual(provider.supportsModeration, false);
    assert.strictEqual(provider.supportsStreaming, true);
    assert.strictEqual(provider.supportsPromptCaching, false);
    assert.ok(provider.systemSuffix.length > 0, 'TranslateGemma should have a system suffix');
    assert.strictEqual(provider.prefixNumber, false);
    assert.strictEqual(provider.lineMatching, false);
  });

  test('model is listed in Ollama', async (t) => {
    skipUnlessReady(t, ollamaReady);

    const client = provider.getClient();
    const models = await client.models.list();
    const ids: string[] = [];
    for await (const m of models) {
      ids.push(m.id);
    }
    assert.ok(
      ids.some((id) => id.startsWith('translategemma')),
      `translategemma not found in models: ${ids.join(', ')}`,
    );
  });
});
