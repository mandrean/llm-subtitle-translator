#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import url from 'url';
import readline from 'node:readline';
import * as undici from 'undici';

import { Command, Option } from 'commander';
import log from 'loglevel';

import {
  DefaultOptions,
  Translator,
  TranslatorStructuredObject,
  TranslatorStructuredArray,
  TranslatorStructuredTimestamp,
  TranslatorAgent,
  OpenAIProvider,
  OllamaProvider,
  OllamaQwen3_32B,
  OllamaTranslateGemma12B,
  OllamaTranslateGemma4B,
  CooldownContext,
  subtitleParser,
  wrapQuotes,
} from '../src/main.js';
import type { TranslationServiceContext, TranslatorOptions, LLMProvider } from '../src/main.js';

import 'dotenv/config';

function getProxyAgent(): undici.EnvHttpProxyAgent | undefined {
  const httpProxyConfig = process.env.http_proxy ?? process.env.HTTP_PROXY;
  const httpsProxyConfig = process.env.https_proxy ?? process.env.HTTPS_PROXY;

  if (httpProxyConfig || httpsProxyConfig) {
    log.debug('[CLI HTTP/HTTPS PROXY]', 'Using HTTP/HTTPS Proxy from ENV Detected', {
      httpProxyConfig,
      httpsProxyConfig,
    });
    return new undici.EnvHttpProxyAgent();
  }

  return undefined;
}

function createProvider(providerName: string, model?: string): LLMProvider {
  const proxyAgent = getProxyAgent();

  switch (providerName) {
    case 'openai':
      return new OpenAIProvider(
        {
          apiKey: process.env.OPENAI_API_KEY ?? '',
          baseURL: process.env.OPENAI_BASE_URL,
          proxyAgent,
        },
        model ?? process.env.OPENAI_DEFAULT_MODEL,
      );
    case 'ollama-qwen3':
      return new OllamaQwen3_32B(process.env.OLLAMA_BASE_URL);
    case 'ollama-translategemma-12b':
      return new OllamaTranslateGemma12B(process.env.OLLAMA_BASE_URL);
    case 'ollama-translategemma-4b':
      return new OllamaTranslateGemma4B(process.env.OLLAMA_BASE_URL);
    case 'ollama': {
      if (!model) throw new Error('--model is required when using the ollama provider');
      return new OllamaProvider({
        model,
        baseURL: process.env.OLLAMA_BASE_URL,
      });
    }
    default:
      throw new Error(`Unknown provider: ${providerName}`);
  }
}

export function createInstance(args: readonly string[]) {
  const program = new Command()
    .description('Translation tool based on LLM APIs')
    .option('--from <language>', 'Source language')
    .option('--to <language>', 'Target language', 'English')
    .addOption(
      new Option('--provider <provider>', 'LLM provider to use')
        .choices(['openai', 'ollama', 'ollama-qwen3', 'ollama-translategemma-12b', 'ollama-translategemma-4b'])
        .default('openai'),
    )
    .option(
      '-m, --model <model>',
      'Model to use for translation',
      process.env.OPENAI_DEFAULT_MODEL ?? DefaultOptions.createChatCompletionRequest.model,
    )
    .option('--moderation-model <model>', 'OpenAI moderation model', DefaultOptions.moderationModel)

    .option('-i, --input <file>', 'Text file name to use as input, .srt or plain text')
    .option('-o, --output <file>', 'Output file name, defaults to be based on input file name')
    .option(
      '-s, --system-instruction <instruction>',
      'Override the prompt system instruction template `Translate ${from} to ${to}` with this plain text',
    )
    .option('-p, --plain-text <text>', 'Only translate this input plain text')

    .option(
      '--experimental-max_token <value>',
      '',
      (val: string) => parseInt(val, 10),
      0,
    )
    .option(
      '--experimental-input-multiplier <value>',
      '',
      (val: string) => parseInt(val, 10),
      0,
    )
    .addOption(
      new Option(
        '-r, --structured <mode>',
        'Structured response format mode',
      )
        .choices(['array', 'timestamp', 'agent', 'object', 'none'])
        .default('array'),
    )
    .option(
      '-c, --context <tokens>',
      'Max context token budget for history',
      (val: string) => parseInt(val, 10),
      DefaultOptions.useFullContext,
    )

    .option(
      '--initial-prompts <prompts>',
      'Initial prompt messages before the translation request messages, as a JSON array',
      JSON.parse,
      DefaultOptions.initialPrompts as any,
    )
    .option('--use-moderator', 'Use the OpenAI Moderation tool')
    .option("--no-prefix-number", "Don't prefix lines with numerical indices")
    .option("--no-line-matching", "Don't enforce one to one line quantity input output matching")
    .option(
      '-b, --batch-sizes <sizes>',
      'Batch sizes for translation prompts in JSON Array',
      JSON.parse,
      DefaultOptions.batchSizes as any,
    )
    .option(
      '-t, --temperature <temperature>',
      'Sampling temperature',
      parseFloat,
      DefaultOptions.createChatCompletionRequest.temperature,
    )
    .option('--no-stream', 'Disable stream progress output to terminal')
    .option('--top_p <top_p>', 'Nucleus sampling parameter', parseFloat)
    .option('--presence_penalty <presence_penalty>', 'Presence penalty', parseFloat)
    .option('--frequency_penalty <frequency_penalty>', 'Frequency penalty', parseFloat)
    .option('--logit_bias <logit_bias>', 'Logit bias', JSON.parse)
    .option('--reasoning_effort <reasoning_effort>', 'Reasoning effort for reasoning models')
    .addOption(
      new Option('--log-level <level>', 'Log level').choices([
        'trace',
        'debug',
        'info',
        'warn',
        'error',
        'silent',
      ]),
    )
    .option('--silent', 'Same as --log-level silent')
    .option('--quiet', 'Same as --log-level silent')
    .parse(args);

  const opts = program.opts();
  const options: Partial<TranslatorOptions> = {
    createChatCompletionRequest: {
      ...(opts.model && { model: opts.model }),
      ...(opts.temperature !== undefined && { temperature: opts.temperature }),
      ...(opts.top_p !== undefined && { top_p: opts.top_p }),
      stream: opts.stream,
      ...(opts.presence_penalty !== undefined && { presence_penalty: opts.presence_penalty }),
      ...(opts.frequency_penalty !== undefined && { frequency_penalty: opts.frequency_penalty }),
      ...(opts.logit_bias && { logit_bias: opts.logit_bias }),
      ...(opts.reasoning_effort && { reasoning_effort: opts.reasoning_effort }),
    },
    ...(opts.initialPrompts && { initialPrompts: opts.initialPrompts }),
    ...(opts.moderationModel !== undefined && { moderationModel: opts.moderationModel }),
    ...(opts.useModerator !== undefined && { useModerator: opts.useModerator }),
    ...(opts.prefixNumber !== undefined && { prefixNumber: opts.prefixNumber }),
    ...(opts.lineMatching !== undefined && { lineMatching: opts.lineMatching }),
    ...(opts.batchSizes && { batchSizes: opts.batchSizes }),
    ...(opts.structured && opts.structured !== 'none' && { structuredMode: opts.structured }),
    ...(opts.experimentalMax_token && { max_token: opts.experimentalMax_token }),
    ...(opts.experimentalInputMultiplier && { inputMultiplier: opts.experimentalInputMultiplier }),
    ...(opts.context !== undefined && { useFullContext: opts.context }),
    ...(opts.logLevel && { logLevel: opts.logLevel }),
    ...(opts.input && { inputFile: opts.input }),
  };

  log.setDefaultLevel('debug');

  if (opts.silent || opts.quiet) {
    options.logLevel = 'silent';
  }

  if (options.logLevel) {
    log.setLevel(options.logLevel);
  }

  log.debug(
    '[CLI]',
    'Log level',
    Object.entries(log.levels).find((x) => x[1] === log.getLevel())?.[0],
  );

  if (options.inputMultiplier && !options.max_token) {
    log.error(
      '[CLI]',
      '[ERROR]',
      '--experimental-input-multiplier must be set with --experimental-max_token',
    );
    process.exit(1);
  }

  return { opts, options };
}

if (import.meta.url === url.pathToFileURL(process.argv[1]).href) {
  const { opts, options } = createInstance(process.argv);

  const provider = createProvider(opts.provider, opts.model);

  // Warn if structured mode is used with a provider that doesn't support it
  if (
    options.structuredMode &&
    options.structuredMode !== 'none' &&
    !provider.supportsStructuredOutput
  ) {
    log.warn(
      '[CLI]',
      `Provider "${provider.name}" does not support structured output. Falling back to plain text mode.`,
    );
    options.structuredMode = false;
  }

  const coolerChatGPTAPI = new CooldownContext(
    Number(process.env.OPENAI_API_RPM ?? 500),
    60000,
    'ChatGPTAPI',
  );
  const coolerOpenAIModerator = new CooldownContext(
    Number(process.env.OPENAI_API_MODERATOR_RPM ?? process.env.OPENAI_API_RPM ?? 500),
    60000,
    'OpenAIModerator',
  );

  const services: TranslationServiceContext = {
    provider,
    cooler: coolerChatGPTAPI,
    onStreamChunk:
      log.getLevel() === log.levels.SILENT
        ? () => {}
        : (data) => {
            return process.stdout.write(data);
          },
    onStreamEnd:
      log.getLevel() === log.levels.SILENT
        ? () => {}
        : () => {
            return process.stdout.write('\n');
          },
    onClearLine:
      log.getLevel() === log.levels.SILENT
        ? () => {}
        : () => {
            readline.clearLine(process.stdout, 0);
            readline.cursorTo(process.stdout, 0);
          },
    ...(provider.supportsModeration && {
      moderationService: {
        openai: provider.getClient(),
        cooler: coolerOpenAIModerator,
      },
    }),
  };

  function getTranslator() {
    if (options.structuredMode == 'array') {
      return new TranslatorStructuredArray(
        { from: opts.from, to: opts.to },
        services,
        options,
      );
    } else if (options.structuredMode == 'timestamp') {
      return new TranslatorStructuredTimestamp(
        { from: opts.from, to: opts.to },
        services,
        options,
      );
    } else if (options.structuredMode == 'agent') {
      return new TranslatorAgent({ from: opts.from, to: opts.to }, services, options);
    } else if (options.structuredMode == 'object') {
      return new TranslatorStructuredObject(
        { from: opts.from, to: opts.to },
        services,
        options,
      );
    } else {
      return new Translator({ from: opts.from, to: opts.to }, services, options);
    }
  }

  const translator = getTranslator();

  if (opts.systemInstruction) {
    translator.systemInstruction = opts.systemInstruction;
  }

  if (opts.plainText) {
    if (opts.structured === 'timestamp' || opts.structured === 'agent') {
      log.error('[CLI]', '--plain-text is not supported in timestamp/agent mode.');
      process.exit(1);
    }
    await translatePlainText(translator as Translator, opts.plainText);
  } else if (opts.input) {
    if (opts.input.endsWith('.srt')) {
      log.debug('[CLI]', 'Assume SRT file', opts.input);
      const text = fs.readFileSync(opts.input, 'utf-8');
      const srtArraySource = subtitleParser.fromSrt(text);
      const fileTag = `${opts.systemInstruction ? 'Custom' : opts.to}`;
      const outputFile = opts.output ? opts.output : `${opts.input}.out_${fileTag}.srt`;

      if (options.structuredMode === 'timestamp' || options.structuredMode === 'agent') {
        log.warn(
          '[CLI]',
          `${options.structuredMode === 'agent' ? 'Agent' : 'Timestamp'} mode: progress resumption is not supported, starting from beginning.`,
        );
        fs.writeFileSync(outputFile, '');
        const timestampSource = srtArraySource.map((e) => ({
          start: e.startTime,
          end: e.endTime,
          text: e.text,
        }));
        let outputId = 1;
        try {
          for await (const srtOut of (
            translator as TranslatorStructuredTimestamp
          ).translateSrtLines(timestampSource)) {
            const entry = {
              id: String(outputId),
              startTime: srtOut.start,
              startSeconds: subtitleParser.timestampToSeconds(srtOut.start),
              endTime: srtOut.end,
              endSeconds: subtitleParser.timestampToSeconds(srtOut.end),
              text: srtOut.text,
            };
            const outSrt = subtitleParser.toSrt([entry]);
            log.info(
              outputId,
              entry.startTime,
              '->',
              entry.endTime,
              wrapQuotes(entry.text),
            );
            await fs.promises.appendFile(outputFile, outSrt);
            outputId++;
          }
        } catch (error) {
          log.error('[CLI]', 'Error', error);
          process.exit(1);
        }
      } else {
        const srtArrayWorking = subtitleParser.fromSrt(text);
        const sourceLines = srtArraySource.map((x) => x.text);
        const progressFile = `${opts.input}.progress_${fileTag}.csv`;

        if (await checkFileExists(progressFile)) {
          const progress = await getProgress(progressFile);

          if (progress.length === sourceLines.length) {
            log.debug('[CLI]', `Progress already completed ${progressFile}`);
            log.debug('[CLI]', `Overwriting ${progressFile}`);
            fs.writeFileSync(progressFile, '');
            log.debug('[CLI]', `Overwriting ${outputFile}`);
            fs.writeFileSync(outputFile, '');
          } else {
            log.debug('[CLI]', `Resuming from ${progressFile}`, progress.length);
            const translatorInstance = translator as Translator;
            const sourceProgress = sourceLines
              .slice(0, progress.length)
              .map((x, i) => translatorInstance.preprocessLine(x, i, 0));
            for (let index = 0; index < progress.length; index++) {
              let transform = progress[index];
              if (transform.startsWith('[Flagged]')) {
                translatorInstance.moderatorFlags.set(index, transform);
              } else {
                transform = translatorInstance.preprocessLine(transform, index, 0);
              }
              translatorInstance.workingProgress.push({
                source: sourceProgress[index],
                transform,
              });
            }
            translatorInstance.offset = progress.length;
          }
        } else {
          fs.writeFileSync(outputFile, '');
        }

        try {
          for await (const output of translator.translateLines(sourceLines)) {
            const csv = `${output.index}, ${wrapQuotes(output.finalTransform.replaceAll('\n', '\\N'))}\n`;
            const srtEntry = srtArrayWorking[output.index - 1];
            srtEntry.text = output.finalTransform;
            const outSrt = subtitleParser.toSrt([srtEntry]);
            log.info(
              output.index,
              wrapQuotes(output.source),
              '->',
              wrapQuotes(output.finalTransform),
            );
            await Promise.all([
              fs.promises.appendFile(progressFile, csv),
              fs.promises.appendFile(outputFile, outSrt),
            ]);
          }
        } catch (error) {
          log.error('[CLI]', 'Error', error);
          process.exit(1);
        }
      }
    } else {
      log.debug('[CLI]', 'Assume plain text file', opts.input);
      const fileTag = `${opts.systemInstruction ? 'Custom' : opts.to}`;
      const ext = path.extname(opts.input);
      const outputFile = opts.output
        ? opts.output
        : `${opts.input}.out_${fileTag}${ext}`;
      const text = fs.readFileSync(opts.input, 'utf-8');
      fs.writeFileSync(outputFile, '');
      await translatePlainText(translator as Translator, text, outputFile);
    }
  }
}

async function translatePlainText(
  translator: Translator,
  text: string,
  outfile?: fs.PathLike,
) {
  const lines = text.split(/\r?\n/);
  if (lines[lines.length - 1].length === 0) {
    lines.pop();
  }
  try {
    for await (const output of translator.translateLines(lines)) {
      if (!translator.options.createChatCompletionRequest.stream) {
        log.info(output.transform);
      }
      if (outfile) {
        fs.appendFileSync(outfile, output.transform + '\n');
      }
    }
  } catch (error) {
    log.error('[CLI]', 'Error', error);
    process.exit(1);
  }
}

async function getProgress(progressFile: string): Promise<string[]> {
  const content = await fs.promises.readFile(progressFile, 'utf-8');
  const lines = content.split(/\r?\n/);
  const progress: string[] = [];

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (line.trim() === '') {
      continue;
    }
    const splits = line.split(',');
    const id = Number(splits[0]);
    const text = splits[1].trim();
    const expectedId = index + 1;
    if (id === expectedId) {
      progress.push(text.substring(1, text.length - 1));
    } else {
      throw `Progress csv file not in order. Expected index ${expectedId}, got ${id}, text: ${text}`;
    }
  }
  return progress;
}

async function checkFileExists(filePath: fs.PathLike): Promise<boolean> {
  try {
    await fs.promises.access(filePath);
    return true;
  } catch {
    return false;
  }
}
