import srtParser2 from 'srt-parser-2';
import log from 'loglevel';

export const parser = new srtParser2();

export function lineLabeler(text: string, label: string): string {
  return `${label}. ${text}`;
}

export function splitStringByNumberLabel(str: string): { number: number | undefined; text: string } {
  const regex = /^(\d+\.)?\s*(.*)/;
  const matches = str.match(regex)!;
  const number = matches[1] ? parseInt(matches[1]) : undefined;
  const text = matches[2].trim();
  return { number, text };
}

function formatTimestamp(totalMs: number): string {
  const h = Math.floor(totalMs / 3600000);
  const m = Math.floor((totalMs % 3600000) / 60000);
  const s = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`;
}

export function millisecondsToTimestamp(milliseconds: number): string {
  return formatTimestamp(milliseconds);
}

export function timestampToMilliseconds(timestamp: string): number {
  const [h, m, rest] = timestamp.split(':');
  const [s, ms] = rest.split(',');
  return parseInt(h) * 3600000 + parseInt(m) * 60000 + parseInt(s) * 1000 + parseInt(ms);
}

export function secondsToTimestamp(seconds: number): string {
  return formatTimestamp(Math.floor(seconds * 1000));
}

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
