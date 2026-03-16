export const genRanHex = (size: number): string =>
  [...Array(size)].map(() => Math.floor(Math.random() * 16).toString(16)).join('');

export async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export interface RetryContext {
  currentTry: number;
  error: any;
  maxRetries: number;
}

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

export function wrapQuotes(text: string): string {
  return `"${text.replaceAll('"', '\\"')}"`;
}

export function roundWithPrecision(num: number, precision: number): number {
  const multiplier = Math.pow(10, precision);
  return Math.round(num * multiplier) / multiplier;
}
