import test from 'node:test';
import assert from 'node:assert';
import { CooldownContext } from '../src/main.js';
import { sleep } from '../src/helpers.js';

function testCooldown(burstCount: number, totalCount: number, waitTime: number) {
  test(`CooldownContext should handle ${burstCount} bursts correctly`, async () => {
    const description = `tester ${burstCount}/${totalCount} ${waitTime}ms`;
    const cooler = new CooldownContext(burstCount, waitTime, description);
    cooler.baseDelay = 15;

    let lastTime = Date.now();

    let workDone = 0;

    for (let index = 0; index < totalCount; index++) {
      const cooled = await cooler.cool();
      if (cooled) {
        const currentTime = Date.now();
        const timeDiff = currentTime - lastTime;
        assert(
          workDone === burstCount,
          `Work done ${workDone} per cool should be same as specified burst count ${burstCount}`,
        );
        assert(
          timeDiff >= waitTime,
          `Expected a wait time of at least ${waitTime} ms between bursts, but got ${timeDiff} ms`,
        );
        lastTime = currentTime;
        workDone = 0;
      }

      await sleep(10);
      workDone++;
    }
  });
}

while (true) {
  testCooldown(1, 5, 33);
  testCooldown(2, 5, 50);
  testCooldown(10, 30, 250);
  break;
}
