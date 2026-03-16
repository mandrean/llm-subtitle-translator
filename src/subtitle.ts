import srtParser2 from 'srt-parser-2';
import log from 'loglevel';

/** Shared SRT parser/serializer instance. */
export const parser = new srtParser2();

/**
 * Prepends a numbered label to a line of text, producing a format like `"1. Hello"`.
 *
 * Used to tag subtitle lines so the LLM can return translations in the same order.
 *
 * @param text - The subtitle text to label.
 * @param label - The label (typically a sequence number) to prepend.
 * @returns The labeled string.
 * @example
 * ```ts
 * lineLabeler('Hello world', '3'); // "3. Hello world"
 * ```
 */
export function lineLabeler(text: string, label: string): string {
  return `${label}. ${text}`;
}

/**
 * Splits a string that may start with a numeric label (e.g. `"3. Hello"`) into its
 * label number and the remaining text.
 *
 * @param str - The input string, optionally prefixed with a number label.
 * @returns An object with the parsed `number` (or `undefined` if no label) and the `text`.
 * @example
 * ```ts
 * splitStringByNumberLabel('3. Hello world');
 * // { number: 3, text: 'Hello world' }
 *
 * splitStringByNumberLabel('No label here');
 * // { number: undefined, text: 'No label here' }
 * ```
 */
export function splitStringByNumberLabel(str: string): { number: number | undefined; text: string } {
  const regex = /^(\d+\.)?\s*(.*)/;
  const matches = str.match(regex)!;
  const number = matches[1] ? parseInt(matches[1]) : undefined;
  const text = matches[2].trim();
  return { number, text };
}

/**
 * Formats a duration in milliseconds into an SRT timestamp string (`HH:MM:SS,mmm`).
 *
 * @param totalMs - Total milliseconds to format.
 * @returns The formatted timestamp string.
 */
function formatTimestamp(totalMs: number): string {
  const h = Math.floor(totalMs / 3600000);
  const m = Math.floor((totalMs % 3600000) / 60000);
  const s = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
}

/**
 * Converts milliseconds to an SRT timestamp string (`HH:MM:SS,mmm`).
 *
 * @param milliseconds - Duration in milliseconds.
 * @returns The formatted SRT timestamp.
 * @example
 * ```ts
 * millisecondsToTimestamp(3723004); // "01:02:03,004"
 * ```
 */
export function millisecondsToTimestamp(milliseconds: number): string {
  return formatTimestamp(milliseconds);
}

/**
 * Parses an SRT timestamp string (`HH:MM:SS,mmm`) into milliseconds.
 *
 * @param timestamp - The SRT timestamp string to parse.
 * @returns The total duration in milliseconds.
 * @example
 * ```ts
 * timestampToMilliseconds('01:02:03,004'); // 3723004
 * ```
 */
export function timestampToMilliseconds(timestamp: string): number {
  const [h, m, rest] = timestamp.split(':');
  const [s, ms] = rest.split(',');
  return parseInt(h) * 3600000 + parseInt(m) * 60000 + parseInt(s) * 1000 + parseInt(ms);
}

/**
 * Converts a duration in seconds to an SRT timestamp string (`HH:MM:SS,mmm`).
 *
 * @param seconds - Duration in seconds (may include a fractional part).
 * @returns The formatted SRT timestamp.
 * @example
 * ```ts
 * secondsToTimestamp(3723.004); // "01:02:03,004"
 * ```
 */
export function secondsToTimestamp(seconds: number): string {
  return formatTimestamp(Math.floor(seconds * 1000));
}

/**
 * Parses a time offset value into seconds.
 *
 * Accepts either a numeric value (returned as-is) or a string in one of these formats:
 * - A plain number (e.g. `"1.5"`) interpreted as seconds.
 * - `HH:MM:SS:mmm` (four colon-separated parts) interpreted as hours, minutes, seconds, and milliseconds.
 * - An optional leading `-` for negative offsets.
 *
 * @param timeOffset - The time offset as a number (seconds) or a string to parse.
 * @returns The offset in seconds (may be negative).
 * @example
 * ```ts
 * parseTimeOffset('01:02:03:500'); // 3723.5
 * parseTimeOffset('-5');           // -5
 * parseTimeOffset(10);             // 10
 * ```
 */
export function parseTimeOffset(timeOffset: string | number): number {
  if (typeof timeOffset === 'string') {
    let negative = false;
    if (timeOffset.startsWith('-')) {
      negative = true;
      timeOffset = timeOffset.substring(1);
    }
    timeOffset = timeOffset.replace(',', '.');
    timeOffset = timeOffset.replace(/[-.]/g, ':');
    const timeParts = timeOffset.split(':');
    let result: number;

    log.debug(timeParts);

    if (timeParts.length === 1) {
      result = parseFloat(timeParts[0]);
    } else if (timeParts.length === 4) {
      const hours = parseInt(timeParts[0]);
      const minutes = parseInt(timeParts[1]);
      const seconds = parseInt(timeParts[2]);
      const subseconds = parseFloat(timeParts[3]) / 1000;
      result = hours * 3600 + minutes * 60 + seconds + (Math.round(subseconds * 1000) / 1000);
    } else {
      result = NaN;
    }
    return negative ? -result : result;
  }
  return timeOffset;
}

/**
 * Shifts all timestamps in an SRT subtitle string by a given number of seconds.
 *
 * Parses the SRT content, adjusts every start and end time, then serializes it back.
 *
 * @param srtString - The raw SRT file content.
 * @param seconds - The number of seconds to shift (positive = forward, negative = backward).
 * @returns The adjusted SRT content as a string.
 */
export function offsetSrt(srtString: string, seconds: number): string {
  const srt = parser.fromSrt(srtString);

  for (const item of srt) {
    item.startSeconds += seconds;
    item.startTime = secondsToTimestamp(item.startSeconds);
    item.endSeconds += seconds;
    item.endTime = secondsToTimestamp(item.endSeconds);
  }

  return parser.toSrt(srt);
}
