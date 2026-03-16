#!/usr/bin/env node
import url from 'node:url';
import fs from 'node:fs';
import { Command } from 'commander';
import path from 'node:path';
import { offsetSrt, parseTimeOffset, parser } from '../src/subtitle.js';

export function createInstance(args: readonly string[]) {
  const commandOffsetFile = new Command('offset')
    .description(
      'Offsets all timestamps in .srt file, currently implemented using floating points, sub-second operations will have precision issues\n' +
        'For negative offsets, pass -- first, eg: \n./subtitle.mjs -- offset file.srt -01:02:03.456',
    )
    .argument('<file>', 'Target file')
    .argument(
      '<offset>',
      'Time offset in HH-MM-SS.sss or HH:MM:SS,sss or HH:MM:SS.sss or seconds',
    )
    .action((file: string, offset: string) => offsetFile(file, offset));

  const commandMergeFiles = new Command('merge')
    .description('Merge subtitle files')
    .arguments('<files...>')
    .action((files: string[]) => mergeFiles(files));

  const program = new Command()
    .description('Subtitle utilities')
    .addCommand(commandOffsetFile)
    .addCommand(commandMergeFiles)
    .parse(args);

  const opts = program.opts();

  return { program, opts };
}

export function offsetFile(file: string, offset: string) {
  const offsetSeconds = parseTimeOffset(offset);

  if (isNaN(offsetSeconds)) {
    console.error('Bad format', offset);
    return;
  }

  const filePath = path.parse(file);

  console.error('offsetting', filePath.ext, filePath.name, offsetSeconds);

  const content = fs.readFileSync(file, 'utf-8');
  const srt = offsetSrt(content, offsetSeconds);

  fs.renameSync(file, path.join(filePath.dir, filePath.name + '.old' + filePath.ext));
  fs.writeFileSync(file, srt);
}

export function mergeFiles(files: string[]) {
  const output: any[] = [];

  for (const file of files) {
    const content = fs.readFileSync(file, 'utf-8');
    const srt = parser.fromSrt(content);
    for (let index = 0, id = output.length; index < srt.length; index++, id++) {
      const item = srt[index];
      item.id = (id + 1).toString();
      output.push(item);
    }
  }
  const outSrt = parser.toSrt(output);
  const outFilePaths = files.map((x) => path.parse(x));
  const outFileName =
    outFilePaths.map((x) => x.name).join('+') + outFilePaths[0].ext;
  fs.writeFileSync(path.join(outFilePaths[0].dir, outFileName), outSrt);
}

if (import.meta.url === url.pathToFileURL(process.argv[1]).href) {
  const { opts } = createInstance(process.argv);
}
