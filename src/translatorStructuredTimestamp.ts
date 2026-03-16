import { PassThrough } from 'stream';
import { z } from 'zod';
import { JSONParser } from '@streamparser/json-node';
import type { OpenAI } from 'openai';
import log from 'loglevel';
import { countTokens } from 'gpt-tokenizer';

import { TranslationOutput } from './translatorOutput.js';
import { TranslatorStructuredBase } from './translatorStructuredBase.js';
import type { TranslationServiceContext, TranslatorOptions } from './translatorBase.js';
import { timestampToMilliseconds, millisecondsToTimestamp } from './subtitle.js';
import { encode as encodeToon } from '@toon-format/toon';

const timestampEntriesSchema = z.array(
  z.object({
    start: z.int(),
    end: z.int(),
    text: z.string(),
  }),
);

export interface TimestampEntry {
  start: string;
  end: string;
  text: string;
}

export type MsEntry = z.infer<typeof timestampEntriesSchema>[number];

export const toMsEntry = (e: TimestampEntry): MsEntry => ({
  start: timestampToMilliseconds(e.start),
  end: timestampToMilliseconds(e.end),
  text: e.text,
});

const fromMsEntry = (e: MsEntry): TimestampEntry => ({
  start: millisecondsToTimestamp(e.start),
  end: millisecondsToTimestamp(e.end),
  text: e.text,
});

const singleTimestampSchema = z.object({
  outputs: timestampEntriesSchema,
});

const batchTimestampSchema = z.object({
  outputs: timestampEntriesSchema,
  remarksIfContainedMergers: z.string(),
});

const schemaDescriptions = {
  single: ['outputs: Subtitle entries with start and end as milliseconds'].join('\n'),
  batch: [
    'outputs: Subtitle entries with start and end as milliseconds',
    'remarksIfContainedMergers: MUST be empty if no merges! Otherwise: briefly explain why entries were merged in 1 short sentence.',
  ].join('\n'),
};

export interface BatchTimestampOutput {
  outputs: TimestampEntry[];
  remarksIfContainedMergers: string;
}

interface EntryHistoryItem {
  input: TimestampEntry;
  output: TimestampEntry;
  completionTokens: number;
}

export class TranslatorStructuredTimestamp extends TranslatorStructuredBase<TimestampEntry> {
  entryHistory: EntryHistoryItem[];
  currentBatchEntries?: TimestampEntry[];

  constructor(
    language: { from?: string; to: string },
    services: TranslationServiceContext,
    options?: Partial<TranslatorOptions>,
  ) {
    const opts = options ?? {};
    if (opts.lineMatching) {
      log.warn(
        '[TranslatorStructuredTimestamp]',
        '--no-line-matching must be used in timestamp mode, overriding.',
      );
    }
    opts.lineMatching = false;
    super(language, services, opts);

    this.entryHistory = [];
  }

  async doTranslatePrompt(entries: TimestampEntry[]): Promise<TranslationOutput> {
    const isSingle = entries.length === 1;
    const schema = isSingle ? singleTimestampSchema : batchTimestampSchema;
    const schemaAppendix = isSingle ? schemaDescriptions.single : schemaDescriptions.batch;
    const userMessage: OpenAI.Chat.ChatCompletionMessageParam = {
      role: 'user',
      content: encodeToon({ inputs: entries.map(toMsEntry) }),
    };
    const systemContent = this.systemInstruction
      ? `${this.systemInstruction}\n\n# Output Schema\n${schemaAppendix}`
      : undefined;
    const systemMessage: OpenAI.Chat.ChatCompletionMessageParam[] = systemContent
      ? [{ role: 'system', content: systemContent }]
      : [];
    const messages = [...systemMessage, ...this.options.initialPrompts, ...this.promptContext, userMessage];
    const max_tokens = this.getMaxToken(entries as any);

    try {
      this.currentBatchEntries = entries;

      await this.services.cooler?.cool();

      const output = await this.streamParse(
        {
          messages,
          ...this.options.createChatCompletionRequest,
          stream: this.options.createChatCompletionRequest.stream,
          max_tokens,
        } as OpenAI.ChatCompletionCreateParams,
        {
          structure: schema,
          name: 'translation_timestamp',
        },
        true,
      );

      const translationCandidate = output.choices[0].message as any;

      const parsedRaw = translationCandidate.refusal ? null : translationCandidate.parsed;
      const parsed = parsedRaw
        ? { ...parsedRaw, outputs: parsedRaw.outputs?.map(fromMsEntry) ?? [] }
        : null;

      return TranslationOutput.fromCompletion(parsed as any, output);
    } catch (error: any) {
      log.error(
        '[TranslatorStructuredTimestamp]',
        `Error ${error?.constructor?.name}`,
        error?.message,
      );
      return this.handleTranslateError(error, entries.length)!;
    }
  }

  buildTimestampContext() {
    if (this.entryHistory.length === 0) return;

    const chunkSize = this.options.batchSizes[this.options.batchSizes.length - 1];

    const allChunks: { userContent: string; assistantContent: string; size: number }[] = [];
    for (let i = 0; i < this.entryHistory.length; i += chunkSize) {
      const chunk = this.entryHistory.slice(i, i + chunkSize);
      const userContent = encodeToon({ inputs: chunk.map((e) => toMsEntry(e.input)) });
      const seenStarts = new Set<string>();
      const outputs = chunk.reduce<TimestampEntry[]>((acc, e) => {
        if (!seenStarts.has(e.output.start)) {
          seenStarts.add(e.output.start);
          acc.push(e.output);
        }
        return acc;
      }, []);
      const assistantContent = JSON.stringify({ outputs: outputs.map(toMsEntry) });
      allChunks.push({ userContent, assistantContent, size: chunk.length });
    }

    const { includedChunks, tokenCount } = this.selectContextChunks(
      allChunks,
      ({ userContent, assistantContent }) =>
        countTokens(userContent) + countTokens(assistantContent),
    );

    if (this.options.useFullContext > 0) {
      const totalEntries = this.entryHistory.length;
      const includedEntries = includedChunks.reduce((sum, c) => sum + c.size, 0);
      const logMsg =
        includedEntries < totalEntries
          ? `sliced ${totalEntries - includedEntries} entries (${includedEntries}/${totalEntries} kept, ${tokenCount} tokens)`
          : `all (${includedEntries} entries, ${tokenCount} tokens)`;
      log.debug('[TranslatorStructuredTimestamp]', 'Context:', logMsg);
    }

    this.promptContext = includedChunks.flatMap(({ userContent, assistantContent }) => [
      { role: 'user' as const, content: userContent },
      { role: 'assistant' as const, content: assistantContent },
    ]);
  }

  async *translateSingleSrt(entries: TimestampEntry[]) {
    log.debug('[TranslatorStructuredTimestamp]', 'Single entry mode');
    for (const entry of entries) {
      this.buildTimestampContext();
      const output = await this.translatePrompt([entry] as any);
      const outputEntries: TimestampEntry[] = (output.content as any)?.outputs ?? [];
      const resultEntry = outputEntries?.[0] ?? entry;

      if (!outputEntries?.[0]) {
        log.warn(
          '[TranslatorStructuredTimestamp]',
          'Empty output for single entry, using original:',
          entry.text,
        );
      }

      this.entryHistory.push({
        input: entry,
        output: resultEntry,
        completionTokens: output.completionTokens,
      });

      yield resultEntry;
    }
  }

  evaluateBatchOutput(
    batch: TimestampEntry[],
    outputEntries: TimestampEntry[],
    remarksIfContainedMergers: string,
  ): boolean {
    const firstInputStart = batch[0].start;
    const firstOutputStart = outputEntries[0]?.start;
    const lastInputEnd = batch.at(-1)!.end;
    const lastOutputEnd = outputEntries.at(-1)?.end;
    const isMismatch =
      outputEntries.length === 0 ||
      firstOutputStart !== firstInputStart ||
      lastOutputEnd !== lastInputEnd;
    const actuallyMerged = outputEntries.length !== batch.length;

    this.logMergeStatus(
      batch,
      outputEntries,
      remarksIfContainedMergers,
      isMismatch,
      actuallyMerged,
      lastInputEnd,
    );

    if (isMismatch) {
      log.debug(
        '[TranslatorStructuredTimestamp]',
        'Timestamp boundary mismatch',
        'expected start:',
        firstInputStart,
        'got:',
        firstOutputStart,
        'expected end:',
        lastInputEnd,
        'got:',
        lastOutputEnd,
        `(input: ${batch.length}, output: ${outputEntries.length})`,
        ...(remarksIfContainedMergers
          ? [`remarks: "${remarksIfContainedMergers}"`]
          : []),
      );
    }

    return isMismatch;
  }

  logMergeStatus(
    batch: TimestampEntry[],
    outputEntries: TimestampEntry[],
    remarksIfContainedMergers: string,
    isMismatch: boolean,
    actuallyMerged: boolean,
    lastInputEnd: string,
  ) {
    const declaredMerged = remarksIfContainedMergers !== '';
    if (!isMismatch && declaredMerged !== actuallyMerged) {
      log.warn(
        '[TranslatorStructuredTimestamp]',
        `Merge remarksIfContainedMergers mismatch: model ${declaredMerged ? `declared merging ("${remarksIfContainedMergers}")` : 'provided no remarks'} but output count ${outputEntries.length} vs input ${batch.length}`,
      );
    }

    if (!isMismatch && actuallyMerged) {
      const mergeIdx = outputEntries.findIndex(
        (o, i) => batch[i] && o.start !== batch[i].start,
      );
      const mergeStart = mergeIdx === -1 ? outputEntries.length : mergeIdx;
      const rangeStart = (batch[mergeStart] ?? batch.at(-1)!).start;

      const outputStartSet = new Set(outputEntries.map((e) => e.start));
      const inputStartSet = new Set(batch.map((e) => e.start));
      const reconcileAt =
        [...outputStartSet]
          .filter((s) => inputStartSet.has(s) && s >= rangeStart)
          .sort()[0] ?? lastInputEnd;

      const fmtEntry = (e: TimestampEntry) => `\n  ${e.start}: "${e.text}"`;
      const inputStartIdx = Math.max(0, mergeStart - 1);
      const inputToLog = batch.slice(inputStartIdx).filter((e) => e.start <= reconcileAt);
      const outputToLog = outputEntries.filter(
        (e) => e.end >= rangeStart && e.start <= reconcileAt,
      );
      log.debug(
        '[TranslatorStructuredTimestamp]',
        'Merging detected',
        `(input: ${batch.length}, output: ${outputEntries.length})`,
        `\n input:${inputToLog.map(fmtEntry).join('')}`,
        `\n output:${outputToLog.map(fmtEntry).join('')}`,
        remarksIfContainedMergers ? `\n remarks: ${remarksIfContainedMergers}` : '',
      );
    }
  }

  async *translateSrtLines(entries: TimestampEntry[]) {
    log.debug(
      '[TranslatorStructuredTimestamp]',
      'System Instruction:',
      this.systemInstruction,
    );
    this.aborted = false;

    for (
      let index = 0, reducedBatchSessions = 0;
      index < entries.length;
      index += this.currentBatchSize
    ) {
      const batch = entries.slice(index, index + this.currentBatchSize);

      this.buildTimestampContext();
      const output = await this.translatePrompt(batch as any);

      if (this.aborted) {
        log.debug('[TranslatorStructuredTimestamp]', 'Aborted');
        return;
      }

      const parsed = (output.content ?? {}) as unknown as BatchTimestampOutput;
      const outputEntries = parsed.outputs ?? [];

      const isMismatch = this.evaluateBatchOutput(
        batch,
        outputEntries,
        parsed.remarksIfContainedMergers ?? '',
      );

      if (isMismatch || (batch.length > 1 && output.refusal)) {
        this.promptTokensWasted += output.promptTokens;
        this.completionTokensWasted += output.completionTokens;

        if (output.refusal) {
          log.debug('[TranslatorStructuredTimestamp]', 'Refusal:', output.refusal);
        }

        if (this.changeBatchSize('decrease')) {
          index -= this.currentBatchSize;
        } else {
          yield* this.translateSingleSrt(batch);
        }
      } else {
        const completionTokensPerEntry = output.completionTokens / batch.length;
        for (const input of batch) {
          const matchedOutput =
            outputEntries.find((o) => o.start <= input.start && o.end >= input.end) ??
            outputEntries.at(-1)!;
          this.entryHistory.push({
            input,
            output: matchedOutput,
            completionTokens: completionTokensPerEntry,
          });
        }

        yield* outputEntries;

        if (
          this.batchSizeThreshold &&
          reducedBatchSessions++ >= this.batchSizeThreshold
        ) {
          reducedBatchSessions = 0;
          const old = this.currentBatchSize;
          this.changeBatchSize('increase');
          index -= this.currentBatchSize - old;
        }
      }

      this.printUsage();
    }
  }

  override jsonStreamParse(runner: any): void {
    const passThroughStream = new PassThrough();

    runner.on('content.delta', (e: any) => {
      passThroughStream.write(e.delta);
    });

    runner.on('content.done', () => {
      passThroughStream.end();
    });

    const prevLen = { start: 0, end: 0, text: 0 };
    let textDone = true;
    let expectedIdx = -1;
    const currentBatchEntries = this.currentBatchEntries ?? [];

    const emitField = (
      fieldKey: 'start' | 'end' | 'text',
      separator: string,
      value: string,
      partial: boolean,
      onComplete?: () => void,
    ) => {
      if (!value) {
        return;
      }
      const delta = value.slice(prevLen[fieldKey]);
      if (delta) {
        this.services.onStreamChunk?.(delta);
      }
      prevLen[fieldKey] = value.length;
      if (!partial) {
        this.services.onStreamChunk?.(separator);
        onComplete?.();
      }
    };

    const pipeline = passThroughStream.pipe(
      new JSONParser({
        paths: ['$.outputs.*.start', '$.outputs.*.end', '$.outputs.*.text'],
        keepStack: false,
        emitPartialTokens: true,
        emitPartialValues: true,
      }),
    );

    pipeline.on(
      'data',
      ({ value, key, partial }: { value: string | number; key: string; partial: boolean }) => {
        try {
          if (key === 'start') {
            if (textDone) {
              prevLen.start = prevLen.end = prevLen.text = 0;
              textDone = false;
              expectedIdx++;
            }
            if (!partial) {
              const expectedStart = currentBatchEntries[expectedIdx]?.start;
              if (
                expectedStart &&
                (value as number) !== timestampToMilliseconds(expectedStart)
              ) {
                this.services.onStreamChunk?.('>>> ');
              }
              emitField(
                'start',
                ' -> ',
                millisecondsToTimestamp(value as number),
                partial,
              );
            }
          } else if (key === 'end') {
            if (!partial)
              emitField('end', '  ', millisecondsToTimestamp(value as number), partial);
          } else if (key === 'text') {
            emitField('text', '\n', value as string, partial, () => {
              textDone = true;
            });
          }
        } catch (err) {
          log.error('[TranslatorStructuredTimestamp]', 'Parsing error:', err);
        }
      },
    );

    pipeline.on('error', (err: Error) => {
      log.error('[TranslatorStructuredTimestamp]', 'stream-json parsing error:', err);
    });
  }
}
