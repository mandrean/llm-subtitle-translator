import path from 'path';
import { z } from 'zod';
import type { OpenAI } from 'openai';
import log from 'loglevel';
import { countTokens } from 'gpt-tokenizer';
import { encode as encodeToon } from '@toon-format/toon';
import {
  TranslatorStructuredTimestamp,
  toMsEntry,
} from './translatorStructuredTimestamp.js';
import type { TimestampEntry, BatchTimestampOutput } from './translatorStructuredTimestamp.js';
import type { TranslationServiceContext, TranslatorOptions } from './translatorBase.js';

const scanBatchSchema = z.object({
  batchSummary: z.string().describe('Translation notes for this scan window.'),
  batchSize: z
    .int()
    .describe(
      'Number of entries from the start of this window to commit as one translation batch.',
    ),
});

const consolidateSchema = z.object({
  consolidatedBatchSummary: z
    .string()
    .describe('Condensed synthesis of the provided batch summaries.'),
});

const finalInstructionSchema = z.object({
  finalInstruction: z
    .string()
    .describe('Final translation system instruction for this subtitle file.'),
});

const overviewSchema = z.object({
  overview: z.string().describe('Brief content overview of the subtitle file.'),
});

const agentInstructionSchema = z.object({
  agentInstruction: z
    .string()
    .describe(
      'Enhanced self-instruction for scanning and translating this subtitle file.',
    ),
});

export interface SliceRange {
  start: number;
  end: number;
}

export interface PlanningResult {
  accumulatedBatchSummary: string;
  customSlices: SliceRange[];
}

const agentInstructionTokenBudget = (useFullContext: number): number =>
  Math.floor(useFullContext / 2);

export class TranslatorAgent extends TranslatorStructuredTimestamp {
  _agentContextChunks: { userContent: string; assistantContent: string; size: number }[];
  planningPromptTokens: number;
  planningCompletionTokens: number;

  constructor(
    language: { from?: string; to: string },
    services: TranslationServiceContext,
    options?: Partial<TranslatorOptions>,
  ) {
    super(language, services, options);
    this._agentContextChunks = [];
    this.planningPromptTokens = 0;
    this.planningCompletionTokens = 0;
  }

  _accumulatePlanningUsage(completion: OpenAI.Chat.ChatCompletion) {
    this.planningPromptTokens += completion?.usage?.prompt_tokens ?? 0;
    this.planningCompletionTokens += completion?.usage?.completion_tokens ?? 0;
  }

  override get usage() {
    const base = super.usage;
    return {
      ...base,
      planningPromptTokens: this.planningPromptTokens,
      planningCompletionTokens: this.planningCompletionTokens,
    };
  }

  override async printUsage() {
    await super.printUsage();
    if (this.planningPromptTokens > 0 || this.planningCompletionTokens > 0) {
      log.debug(
        `[TranslatorAgent] Planning tokens:`,
        this.planningPromptTokens,
        '+',
        this.planningCompletionTokens,
        '=',
        this.planningPromptTokens + this.planningCompletionTokens,
      );
    }
  }

  _recordContextChunk(batch: TimestampEntry[], outputEntries: TimestampEntry[]) {
    const userContent = encodeToon({ inputs: batch.map(toMsEntry) });
    const assistantContent = JSON.stringify({ outputs: outputEntries.map(toMsEntry) });
    this._agentContextChunks.push({
      userContent,
      assistantContent,
      size: batch.length,
    });
  }

  override buildTimestampContext() {
    if (this._agentContextChunks.length === 0) return;

    const { includedChunks, tokenCount } = this.selectContextChunks(
      this._agentContextChunks,
      ({ userContent, assistantContent }) =>
        countTokens(userContent) + countTokens(assistantContent),
    );

    if (this.options.useFullContext > 0) {
      const totalEntries = this._agentContextChunks.reduce((s, c) => s + c.size, 0);
      const includedEntries = includedChunks.reduce((s, c) => s + c.size, 0);
      const logMsg =
        includedEntries < totalEntries
          ? `sliced ${totalEntries - includedEntries} entries (${includedEntries}/${totalEntries} kept, ${tokenCount} tokens)`
          : `all (${includedEntries} entries, ${tokenCount} tokens)`;
      log.debug('[TranslatorAgent]', 'Context:', logMsg);
    }

    this.promptContext = includedChunks.flatMap(
      ({ userContent, assistantContent }) => [
        { role: 'user' as const, content: userContent },
        { role: 'assistant' as const, content: assistantContent },
      ],
    );
  }

  override async *translateSingleSrt(entries: TimestampEntry[]) {
    log.debug('[TranslatorAgent]', 'Single entry mode');
    for (const entry of entries) {
      this.buildTimestampContext();
      const output = await this.translatePrompt([entry] as any);
      const outputEntries: TimestampEntry[] = (output.content as any)?.outputs ?? [];
      const resultEntry = outputEntries?.[0] ?? entry;

      if (!outputEntries?.[0]) {
        log.warn(
          '[TranslatorAgent]',
          'Empty output for single entry, using original:',
          entry.text,
        );
      }

      this._recordContextChunk([entry], [resultEntry]);
      this.entryHistory.push({
        input: entry,
        output: resultEntry,
        completionTokens: output.completionTokens,
      });

      yield resultEntry;
    }
  }

  async _runOverviewPass(
    entries: TimestampEntry[],
    subtitleMeta: string,
  ): Promise<{ overview: string; agentInstruction: string } | null> {
    const sampleSize = this.options.batchSizes[0];
    const head = entries.slice(0, sampleSize);
    const tail = entries.length > sampleSize ? entries.slice(-sampleSize) : [];

    const sampleLabel =
      tail.length > 0
        ? `First ${head.length} entries (0-${head.length - 1}) and last ${tail.length} entries (${entries.length - tail.length}-${entries.length - 1})`
        : `All ${head.length} entries`;

    log.debug('[TranslatorAgent]', 'Pass 0 (Overview): sampling', sampleLabel);

    const overview = await this._generateOverview(head, tail, sampleLabel, subtitleMeta);
    if (!overview) return null;

    log.debug('[TranslatorAgent]', 'Overview:', overview);

    const agentInstruction = await this._generateAgentInstruction(overview);
    if (!agentInstruction) return { overview, agentInstruction: '' };

    log.debug('[TranslatorAgent]', 'Agent instruction:', agentInstruction);
    return { overview, agentInstruction };
  }

  async _generateOverview(
    head: TimestampEntry[],
    tail: TimestampEntry[],
    sampleLabel: string,
    subtitleMeta: string,
  ): Promise<string | null> {
    const sampledContent =
      tail.length > 0
        ? `${sampleLabel}:\n\nFirst:\n${encodeToon({ inputs: head.map(toMsEntry) })}\n\nLast:\n${encodeToon({ inputs: tail.map(toMsEntry) })}`
        : `${sampleLabel}:\n${encodeToon({ inputs: head.map(toMsEntry) })}`;
    const userContent = `${subtitleMeta}\n\n${sampledContent}`;

    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content:
          `${this.systemInstruction}\n---\n` +
          `You are previewing a subtitle file before translation. ` +
          `Analyze the subtitle metadata and sampled entries to produce a brief overview (2-5 sentences).\n\n` +
          `Rules:\n` +
          `1. Cover: file name/episode identity, total duration, genre, setting, tone, people names, and any notable linguistic features (dialect, slang, technical jargon).\n` +
          `2. Retain key subtitle metadata (file name, entry count, duration) in the overview.`,
      },
      ...this.options.initialPrompts,
      { role: 'user', content: userContent },
    ];

    try {
      await this.services.cooler?.cool();
      const output = await this.streamParse(
        {
          messages,
          ...this.options.createChatCompletionRequest,
          stream: this.options.createChatCompletionRequest.stream,
          max_tokens: undefined,
        } as OpenAI.ChatCompletionCreateParams,
        { structure: overviewSchema, name: 'agent_overview' },
      );

      this._accumulatePlanningUsage(output);
      const parsed = (output.choices[0]?.message as any)?.parsed;
      if (!parsed || (output.choices[0]?.message as any)?.refusal) {
        log.warn('[TranslatorAgent]', 'Overview step refusal or empty response');
        return null;
      }
      return parsed.overview;
    } catch (error: any) {
      log.warn(
        '[TranslatorAgent]',
        'Overview step failed:',
        error?.message,
        '- continuing without overview',
      );
      return null;
    }
  }

  async _generateAgentInstruction(overview: string): Promise<string | null> {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content:
          `${this.systemInstruction}\n---\n` +
          `Based on the content overview below, produce an enhanced instruction ` +
          `for yourself to use when scanning and translating this subtitle file (3-6 sentences of direct translator guidance).\n\n` +
          `Rules:\n` +
          `1. Carry forward useful metadata from the overview (file identity, duration, entry count, names, genre/tone).\n` +
          `2. Specify what to watch for: scene boundaries, speaker changes, contextual dependencies, terminology consistency, tone/register shifts.`,
      },
      ...this.options.initialPrompts,
      { role: 'user', content: `# Content overview:\n${overview}` },
    ];

    try {
      await this.services.cooler?.cool();
      const output = await this.streamParse(
        {
          messages,
          ...this.options.createChatCompletionRequest,
          stream: this.options.createChatCompletionRequest.stream,
          max_tokens: undefined,
        } as OpenAI.ChatCompletionCreateParams,
        { structure: agentInstructionSchema, name: 'agent_instruction' },
      );

      this._accumulatePlanningUsage(output);
      const parsed = (output.choices[0]?.message as any)?.parsed;
      if (!parsed || (output.choices[0]?.message as any)?.refusal) {
        log.warn(
          '[TranslatorAgent]',
          'Agent instruction step refusal or empty response',
        );
        return null;
      }
      return parsed.agentInstruction;
    } catch (error: any) {
      log.warn(
        '[TranslatorAgent]',
        'Agent instruction step failed:',
        error?.message,
        '- continuing without',
      );
      return null;
    }
  }

  async runPlanningPass(
    entries: TimestampEntry[],
    overviewResult?: { overview: string; agentInstruction: string } | null,
    subtitleMeta?: string,
  ): Promise<PlanningResult> {
    const scanBatchSize =
      this.options.batchSizes[this.options.batchSizes.length - 1];
    const budget = agentInstructionTokenBudget(this.options.useFullContext);
    const rawSlices: SliceRange[] = [];

    log.debug(
      '[TranslatorAgent]',
      'Pass 1 (Planning): scanning',
      entries.length,
      'entries in batches of',
      scanBatchSize,
      '| instruction budget:',
      budget,
      'tokens',
    );

    let accumulatedBatchSummary = '';

    let windowStart = 0;

    for (
      let batchStart = 0;
      batchStart < entries.length;
      batchStart = windowStart
    ) {
      const batch = entries.slice(batchStart, batchStart + scanBatchSize);
      const batchEnd = batchStart + batch.length - 1;

      log.debug('[TranslatorAgent]', 'Scanning window', batchStart, '-', batchEnd);

      await this.services.cooler?.cool();

      let batchSlices: SliceRange[] = [];
      let scanOk = false;

      try {
        const result = await this._runScanBatch(
          batch,
          batchStart,
          accumulatedBatchSummary,
          overviewResult?.agentInstruction,
        );
        if (result) {
          const newNote = result.batchSummary?.trim();
          if (newNote) {
            log.debug(
              '[TranslatorAgent]',
              'Batch summary from window',
              batchStart,
              `(${countTokens(newNote)} tokens):\n`,
              newNote,
            );
            const candidate = accumulatedBatchSummary
              ? `${accumulatedBatchSummary}\n${newNote}`
              : newNote;

            const accumulatedBatchSummaryTokens = countTokens(candidate);
            log.debug(
              '[TranslatorAgent]',
              'Accumulated batch summary length:',
              candidate.length,
              `(${accumulatedBatchSummaryTokens}/${budget} tokens)`,
            );
            if (accumulatedBatchSummaryTokens > budget) {
              log.debug(
                '[TranslatorAgent]',
                'Batch summary accumulator over budget (>',
                budget,
                'tokens) - consolidating',
              );
              await this.services.cooler?.cool();
              accumulatedBatchSummary = await this._consolidateBatchSummaries(
                accumulatedBatchSummary,
                newNote,
                budget,
              );
              log.debug(
                '[TranslatorAgent]',
                'Consolidated batch summary:',
                accumulatedBatchSummary,
              );
              log.debug(
                '[TranslatorAgent]',
                'Consolidated batch summary length:',
                accumulatedBatchSummary.length,
                `(${countTokens(accumulatedBatchSummary)} tokens)`,
              );
            } else {
              accumulatedBatchSummary = candidate;
            }
          }
          const size = Math.max(
            1,
            Math.min(result.batchSize ?? batch.length, batch.length),
          );
          batchSlices = [{ start: batchStart, end: batchStart + size - 1 }];
          scanOk = true;
        } else {
          log.warn(
            '[TranslatorAgent]',
            'Scan batch returned null, using default single slice for range',
            batchStart,
            '-',
            batchEnd,
          );
        }
      } catch (error: any) {
        log.warn(
          '[TranslatorAgent]',
          'Scan batch error:',
          error?.message,
          '- falling back to default slice for range',
          batchStart,
          '-',
          batchEnd,
        );
      }

      if (!scanOk || batchSlices.length === 0) {
        rawSlices.push({ start: batchStart, end: batchEnd });
        windowStart = batchEnd + 1;
      } else {
        const slice = batchSlices[0];
        const sliceSize = slice.end - slice.start + 1;
        const minProgress = Math.max(1, Math.floor(scanBatchSize / 2));

        if (sliceSize < minProgress) {
          log.warn(
            '[TranslatorAgent]',
            'batchSize',
            sliceSize,
            'below minimum progress',
            minProgress,
            '- forcing full window coverage',
          );
          rawSlices.push({ start: batchStart, end: batchEnd });
          windowStart = batchEnd + 1;
        } else {
          rawSlices.push(slice);
          windowStart = slice.end + 1;
          log.debug(
            '[TranslatorAgent]',
            'Committed slice',
            slice.start,
            '-',
            slice.end,
            '| next window starts at',
            windowStart,
          );
        }
      }
    }

    const customSlices = this._normalizeSlices(rawSlices, entries.length);

    let finalInstruction = accumulatedBatchSummary;
    if (accumulatedBatchSummary) {
      await this.services.cooler?.cool();
      const consolidatedContextSummary = await this._consolidateBatchSummaries(
        accumulatedBatchSummary,
        undefined,
        budget,
      );
      log.debug(
        '[TranslatorAgent]',
        'Context summary:\n',
        consolidatedContextSummary,
      );

      await this.services.cooler?.cool();
      const refinedDirective = await this._refineFinalInstruction(
        consolidatedContextSummary,
        budget,
      );
      log.debug(
        '[TranslatorAgent]',
        'Refined instruction:\n',
        refinedDirective,
      );

      finalInstruction = [
        refinedDirective ?? this.systemInstruction,
        consolidatedContextSummary,
        subtitleMeta,
      ]
        .filter(Boolean)
        .join('\n---\n');
    }

    log.debug(
      '[TranslatorAgent]',
      'Pass 1 complete.',
      'Final instruction length:',
      finalInstruction.length,
      `(${countTokens(finalInstruction)} tokens)`,
      '| Custom slices:',
      customSlices.length,
      '| Batch sizes:',
      customSlices.map((s) => s.end - s.start + 1).join(', '),
    );

    return { accumulatedBatchSummary: finalInstruction, customSlices };
  }

  _normalizeSlices(slices: SliceRange[], totalEntries: number): SliceRange[] {
    if (slices.length === 0) return [];

    const sorted = [...slices].sort((a, b) => a.start - b.start);

    const merged = [sorted[0]];
    for (let i = 1; i < sorted.length; i++) {
      const prev = merged.at(-1)!;
      const cur = sorted[i];
      if (cur.start <= prev.end) {
        if (cur.end > prev.end) {
          prev.end = cur.end;
          log.debug(
            '[TranslatorAgent]',
            'Merged overlapping slices at',
            cur.start,
          );
        }
      } else if (cur.start > prev.end + 1) {
        log.debug(
          '[TranslatorAgent]',
          'Gap between slices',
          prev.end,
          'and',
          cur.start,
          '- inserting filler slice',
        );
        merged.push({ start: prev.end + 1, end: cur.start - 1 });
        merged.push(cur);
      } else {
        merged.push(cur);
      }
    }

    if (merged[0].start > 0) {
      log.debug(
        '[TranslatorAgent]',
        'Extending first slice to cover entries 0 -',
        merged[0].start - 1,
      );
      merged[0].start = 0;
    }

    if (merged.at(-1)!.end < totalEntries - 1) {
      merged.at(-1)!.end = totalEntries - 1;
      log.debug(
        '[TranslatorAgent]',
        'Extended last slice to cover remaining entries up to',
        totalEntries - 1,
      );
    }

    return merged;
  }

  async _runScanBatch(
    batch: TimestampEntry[],
    batchStart: number,
    accumulatedBatchSummary: string,
    agentInstruction?: string,
  ): Promise<{ batchSummary: string; batchSize: number } | null> {
    const contextSection = accumulatedBatchSummary
      ? `\n---\nContext from previous segments:\n${accumulatedBatchSummary}\n---\n`
      : '\n---\n';
    const agentSection = agentInstruction
      ? `Scan guidance:\n${agentInstruction}\n\n`
      : '';
    const systemContent = [
      this.systemInstruction,
      contextSection,
      agentSection,
      `You are scanning a context window of entries ${batchStart} to ${batchStart + batch.length - 1} ` +
        `(timestamps ${batch[0].start}–${batch[batch.length - 1].end} ms).\n\n` +
        `Rules for batchSummary:\n` +
        `1. Write in the translated language.\n` +
        `2. Open with your overall impression of this window's content.\n` +
        `3. Write only what is new or notable here - do not repeat or refine prior context.\n` +
        `4. Cover the 5W1H: who (names, roles, relationships), what (events, terms, objects), ` +
        `where (locations, settings), when (time context), why/how (tone, register, dialect, intent).\n\n` +
        `Rules for batchSize:\n` +
        `1. Decide how many entries from the start of this window (position ${batchStart}) ` +
        `to commit as one translation batch - this sets where the next scan window begins.\n` +
        `2. Must be between 1 and ${batch.length} (the current window size).\n` +
        `3. Use the full window size if entries flow naturally together, ` +
        `or a smaller number to break at a scene or topic boundary.`,
    ].join('');

    const userMessage: OpenAI.Chat.ChatCompletionMessageParam = {
      role: 'user',
      content: encodeToon({ inputs: batch.map(toMsEntry) }),
    };
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      { role: 'system', content: systemContent },
      ...this.options.initialPrompts,
      userMessage,
    ];

    const output = await this.streamParse(
      {
        messages,
        ...this.options.createChatCompletionRequest,
        stream: this.options.createChatCompletionRequest.stream,
        max_tokens: undefined,
      } as OpenAI.ChatCompletionCreateParams,
      {
        structure: scanBatchSchema,
        name: 'agent_scan',
      },
    );

    this._accumulatePlanningUsage(output);
    const message = output.choices[0]?.message as any;
    if (!message || message.refusal) {
      log.warn(
        '[TranslatorAgent]',
        'Scan batch refusal or empty response at position',
        batchStart,
      );
      return null;
    }

    const parsed = message.parsed;
    if (parsed?.batchSize != null) {
      parsed.batchSize = Math.max(1, Math.min(parsed.batchSize, batch.length));
    }
    return parsed;
  }

  async _consolidateBatchSummaries(
    existing: string,
    newNote: string = '',
    budget: number,
    budgetFactor: number = 0.5,
  ): Promise<string> {
    const isFinal = !newNote;
    const targetTokens = Math.floor(budget * budgetFactor);
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content:
          (isFinal
            ? `You are doing a final consolidation of all batch summaries for a subtitle file ` +
              `into a single complete set of notes (target: ~${targetTokens} tokens). ` +
              `This will be used as the full context for the subtitles - preserve all details.`
            : `You are doing consolidation of all given batch summary windows for a subtitle file ` +
              `into a single condensed set of notes (target: ~${targetTokens} tokens). ` +
              `More batches will follow - stay concise but keep all unique facts.`) +
          `\n\nRules:\n` +
          `1. Write in the translated language.\n` +
          `2. Open with your overall impression of the content so far.\n` +
          `3. Preserve all unique 5W1H facts (who, what, where, when, why/how - names, locations, terms, tone, dialect).\n` +
          `4. Remove duplicate or contradictory information. ${isFinal ? 'Be thorough - this is the last pass.' : 'Keep it concise - more content is coming.'}`,
      },
      {
        role: 'user',
        content: `Batch summaries:\n${existing}`,
      },
    ];
    try {
      const output = await this.streamParse(
        {
          messages,
          ...this.options.createChatCompletionRequest,
          stream: this.options.createChatCompletionRequest.stream,
          max_tokens: Math.floor(targetTokens * 1.5),
        } as OpenAI.ChatCompletionCreateParams,
        { structure: consolidateSchema, name: 'agent_consolidate' },
      );

      this._accumulatePlanningUsage(output);
      const result = (output.choices[0]?.message as any)?.parsed
        ?.consolidatedBatchSummary?.trim();
      if (result) return result;
    } catch (error: any) {
      log.warn(
        '[TranslatorAgent]',
        'Consolidation failed:',
        error?.message,
        '- truncating',
      );
    }
    const combined = `${existing}\n${newNote}`;
    if (countTokens(combined) <= budget) return combined;
    const existingLines = existing.split('\n');
    for (let drop = 1; drop < existingLines.length; drop++) {
      const trimmed = existingLines.slice(drop).join('\n') + '\n' + newNote;
      if (countTokens(trimmed) <= budget) return trimmed;
    }
    return newNote;
  }

  async _refineFinalInstruction(
    contextSummary: string,
    budget: number,
  ): Promise<string | null> {
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content:
          `You are a professional translation assistant producing a final translation instruction for a subtitle file. ` +
          `Your goal is a lean, focused instruction.\n\n` +
          `Rules:\n` +
          `1. Preserve the target language and any stylistic directives that apply to the observed content.\n` +
          `2. If the base instruction contains a glossary, dictionary, or list of terms/names, ` +
          `filter it to only entries that appear in or are directly relevant to the observed content. ` +
          `Remove any entries not encountered in this file.\n` +
          `3. Remove instructions that are redundant, contradicted, or clearly out of scope given what was observed.\n` +
          `4. Do not embed narrative facts from the context - keep it as clean translator guidance.\n`,
      },
      {
        role: 'user',
        content:
          `# Base instruction:\n${this.systemInstruction}\n\n` +
          `# Observed content context:\n${contextSummary}`,
      },
    ];
    try {
      const output = await this.streamParse(
        {
          messages,
          ...this.options.createChatCompletionRequest,
          stream: this.options.createChatCompletionRequest.stream,
          max_tokens: budget,
        } as OpenAI.ChatCompletionCreateParams,
        { structure: finalInstructionSchema, name: 'agent_refine_instruction' },
      );

      this._accumulatePlanningUsage(output);
      const parsed = (output.choices[0]?.message as any)?.parsed;
      if (parsed?.finalInstruction) return parsed.finalInstruction;
    } catch (error: any) {
      log.warn(
        '[TranslatorAgent]',
        'Final instruction refinement failed:',
        error?.message,
        '- using base instruction',
      );
    }
    return null;
  }

  override async *translateSrtLines(entries: TimestampEntry[]) {
    log.debug(
      '[TranslatorAgent]',
      'Starting agentic 2-pass translation,',
      entries.length,
      'total entries',
    );

    const fileParts: string[] = [];
    if (this.options.inputFile)
      fileParts.push(`File: ${path.basename(this.options.inputFile)}`);
    fileParts.push(`Total entries: ${entries.length}`);
    if (entries.length > 0)
      fileParts.push(`Duration: ${entries[0].start} -> ${entries.at(-1)!.end}`);
    const subtitleMeta = fileParts.join(' | ');

    const baseInstruction = this.systemInstruction;

    const overviewResult = await this._runOverviewPass(entries, subtitleMeta);

    const { accumulatedBatchSummary, customSlices } = await this.runPlanningPass(
      entries,
      overviewResult,
      subtitleMeta,
    );

    this.systemInstruction = accumulatedBatchSummary || baseInstruction;

    if (accumulatedBatchSummary) {
      log.debug(
        '[TranslatorAgent]',
        `System instruction updated:\n${this.systemInstruction}`,
      );
    }

    if (customSlices.length === 0) {
      log.warn(
        '[TranslatorAgent]',
        'Pass 1 produced no slices - falling back to standard translateSrtLines',
      );
      yield* super.translateSrtLines(entries);
      return;
    }

    log.debug(
      '[TranslatorAgent]',
      'Pass 2 (Translation): using',
      customSlices.length,
      'custom slices',
    );
    yield* this._translateWithCustomSlices(entries, customSlices);
  }

  async *_translateWithCustomSlices(
    entries: TimestampEntry[],
    customSlices: SliceRange[],
  ) {
    this.aborted = false;
    let sliceIndex = 0;

    while (sliceIndex < customSlices.length) {
      const slice = customSlices[sliceIndex];
      const batch = entries.slice(slice.start, slice.end + 1);

      if (batch.length === 0) {
        log.warn(
          '[TranslatorAgent]',
          'Empty batch for slice',
          slice,
          '- skipping',
        );
        sliceIndex++;
        continue;
      }

      this.buildTimestampContext();
      const output = await this.translatePrompt(batch as any);

      if (this.aborted) {
        log.debug('[TranslatorAgent]', 'Aborted');
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
          log.debug('[TranslatorAgent]', 'Refusal on slice', slice);
        }

        const nextSize = [...this.options.batchSizes]
          .reverse()
          .find((s) => s < batch.length);
        if (nextSize !== undefined) {
          const subSlices: SliceRange[] = [];
          for (let i = slice.start; i <= slice.end; i += nextSize) {
            subSlices.push({
              start: i,
              end: Math.min(i + nextSize - 1, slice.end),
            });
          }
          customSlices.splice(sliceIndex, 1, ...subSlices);
          log.debug(
            '[TranslatorAgent]',
            'Mismatch on slice',
            slice,
            '- re-splitting into',
            subSlices.length,
            'sub-slices of size',
            nextSize,
          );
        } else {
          yield* this.translateSingleSrt(batch);
          sliceIndex++;
        }
      } else {
        const completionTokensPerEntry = output.completionTokens / batch.length;
        for (const input of batch) {
          const exactMatch = outputEntries.find(
            (o) => o.start <= input.start && o.end >= input.end,
          );
          if (!exactMatch) {
            log.warn(
              '[TranslatorAgent]',
              'No output entry covers input',
              input.start,
              '-',
              input.end,
              `"${input.text}" - using last output as fallback`,
            );
          }
          const matchedOutput = exactMatch ?? outputEntries.at(-1)!;
          this.entryHistory.push({
            input,
            output: matchedOutput,
            completionTokens: completionTokensPerEntry,
          });
        }

        this._recordContextChunk(batch, outputEntries);
        yield* outputEntries;
        sliceIndex++;
      }

      this.printUsage();
    }
  }
}
