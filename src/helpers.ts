/**
 * Generates a random hexadecimal string of the specified length.
 *
 * @param size - The number of hex characters to generate.
 * @returns A random hex string of the given length.
 * @example
 * ```ts
 * genRanHex(8); // e.g. "a3f1b2c9"
 * ```
 */
export const genRanHex = (size: number): string =>
  [...Array(size)].map(() => Math.floor(Math.random() * 16).toString(16)).join('');

/**
 * Returns a promise that resolves after a specified number of milliseconds.
 *
 * @param ms - The number of milliseconds to sleep.
 * @returns A promise that resolves after the delay.
 * @example
 * ```ts
 * await sleep(2000); // pauses execution for 2 seconds
 * ```
 */
export async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Context object passed to retry and failure callbacks in {@link retryWrapper}.
 */
export interface RetryContext {
  /** The current attempt number (1-based). */
  currentTry: number;
  /** The error thrown by the most recent attempt. */
  error: any;
  /** The maximum number of retry attempts allowed. */
  maxRetries: number;
}

/**
 * Executes an async function with automatic retry logic.
 *
 * Calls {@link func} up to {@link maxRetries} times. On each failure the optional
 * {@link onRetry} callback is invoked (useful for logging or adding delays). If all
 * attempts are exhausted, the optional {@link onFail} callback is invoked.
 *
 * @typeParam T - The resolved type of the async function.
 * @param func - The async function to execute.
 * @param maxRetries - Maximum number of attempts before giving up.
 * @param onRetry - Optional callback invoked after each failed attempt.
 * @param onFail - Optional callback invoked when all retries are exhausted.
 * @returns The result of {@link func}, or `undefined` if all retries fail and {@link onFail} does not throw.
 * @example
 * ```ts
 * const result = await retryWrapper(
 *   () => fetchData(),
 *   3,
 *   async (ctx) => console.log(`Retry ${ctx.currentTry}/${ctx.maxRetries}`),
 * );
 * ```
 */
export async function retryWrapper<T>(
  func: () => Promise<T>,
  maxRetries: number,
  onRetry?: (retryContext: RetryContext) => Promise<void>,
  onFail?: (retryContext: RetryContext) => Promise<void>,
): Promise<T | undefined> {
  const retryContext: RetryContext = { currentTry: 1, error: undefined, maxRetries };

  while (retryContext.currentTry <= retryContext.maxRetries) {
    try {
      return await func();
    } catch (error) {
      retryContext.error = error;
      if (onRetry) {
        await onRetry(retryContext);
      }
    }
    retryContext.currentTry++;
  }

  if (onFail) {
    await onFail(retryContext);
  }

  return undefined;
}

/**
 * Wraps a string in double quotes, escaping any existing double quotes within.
 *
 * @param text - The string to wrap.
 * @returns The quoted string with internal double quotes escaped.
 * @example
 * ```ts
 * wrapQuotes('hello "world"'); // '"hello \\"world\\""'
 * ```
 */
export function wrapQuotes(text: string): string {
  return `"${text.replaceAll('"', '\\"')}"`;
}

/**
 * Rounds a number to a specified number of decimal places.
 *
 * @param num - The number to round.
 * @param precision - The number of decimal places.
 * @returns The rounded number.
 * @example
 * ```ts
 * roundWithPrecision(3.14159, 2); // 3.14
 * ```
 */
export function roundWithPrecision(num: number, precision: number): number {
  const multiplier = Math.pow(10, precision);
  return Math.round(num * multiplier) / multiplier;
}
