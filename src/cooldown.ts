import log from 'loglevel';
import { sleep } from './helpers.js';

/**
 * Rate-limiter that tracks requests within a sliding time window and
 * enforces a cooldown period when the limit is exceeded.
 *
 * @example
 * ```ts
 * const cooler = new CooldownContext(5, 60_000, 'API calls');
 * await cooler.cool(); // sleeps if rate limit is reached
 * ```
 */
export class CooldownContext {
  /** Maximum number of requests allowed within the duration window. */
  limit: number;
  /** Length of the sliding window in milliseconds. */
  duration: number;
  /** Human-readable label used in log messages. */
  description: string;
  /** Fixed base delay (in ms) added after the computed cooldown. */
  baseDelay: number;
  /** Timestamps (ms since epoch) of recent requests inside the window. */
  requests: number[];
  /** Number of requests currently within the sliding window. */
  rate: number;

  /**
   * Creates a new cooldown context.
   *
   * @param limit - Maximum number of requests allowed within {@link duration}.
   * @param duration - Sliding window duration in milliseconds.
   * @param description - Label for log output.
   */
  constructor(limit: number, duration: number, description: string) {
    this.limit = limit;
    this.duration = duration;
    this.description = description;
    this.baseDelay = 1000;
    this.requests = [];
    this.rate = 0;
  }

  /**
   * Calculates the remaining cooldown time in milliseconds.
   *
   * Prunes expired request timestamps, updates {@link rate}, and returns how
   * long the caller must wait before the next request is allowed.
   *
   * @returns Milliseconds to wait, or `0` if no cooldown is needed.
   */
  cooldown(): number {
    const now = Date.now();
    this.requests = this.requests.filter(time => now - time < this.duration);
    this.rate = this.requests.length;

    if (this.rate >= this.limit) {
      const nextRequestTime = this.requests[0] + this.duration;
      return nextRequestTime - now;
    }

    return 0;
  }

  /**
   * Enforces the rate limit by sleeping if the cooldown threshold is exceeded.
   *
   * Records the current request timestamp after the cooldown (if any) has elapsed.
   *
   * @returns `true` if the caller was throttled (had to wait), `false` otherwise.
   */
  async cool(): Promise<boolean> {
    const cooldown = this.cooldown();

    if (cooldown === 0) {
      this.requests.push(Date.now());
      return false;
    }
    log.error('[Cooldown]', this.description, cooldown, 'ms');

    await sleep(cooldown + this.baseDelay);
    this.requests.push(Date.now());
    return true;
  }
}
