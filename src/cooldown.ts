import log from 'loglevel';
import { sleep } from './helpers.js';

export class CooldownContext {
  limit: number;
  duration: number;
  description: string;
  baseDelay: number;
  requests: number[];
  rate: number;

  constructor(limit: number, duration: number, description: string) {
    this.limit = limit;
    this.duration = duration;
    this.description = description;
    this.baseDelay = 1000;
    this.requests = [];
    this.rate = 0;
  }

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
