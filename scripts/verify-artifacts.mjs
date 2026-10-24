#!/usr/bin/env node
// Verifies SHA-256 checksums and basic structure of every bundled artifact.
// Run in CI before building so a corrupted or hand-edited artifact can never ship.
import { createHash } from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'public');
const errors = [];
let checked = 0;

const sha = (rel) =>
  createHash('sha256')
    .update(readFileSync(join(root, rel)))
    .digest('hex');
const json = (rel) => JSON.parse(readFileSync(join(root, rel), 'utf8'));

function check(rel, expected) {
  checked++;
  if (!existsSync(join(root, rel))) return errors.push(`missing ${rel}`);
  const actual = sha(rel);
  if (actual !== expected) errors.push(`${rel}: sha256 ${actual} != manifest ${expected}`);
}

const demos = json('demos/manifest.json');
for (const d of demos.demos) {
  check(`demos/${d.file}`, d.sha256);
  check(`demos/${d.attentionFile}`, d.attentionSha256);
  const art = json(`demos/${d.file}`);
  const { layers, heads, queryLength, keyLength } = art.attention.shape;
  const bytes = readFileSync(join(root, 'demos', d.attentionFile)).byteLength;
  if (bytes !== layers * heads * queryLength * keyLength * 4)
    errors.push(`${d.attentionFile}: size does not match shape`);
  if (art.input.tokens.length !== queryLength)
    errors.push(`${d.file}: token count does not match tensor shape`);
  if (art.attention.sha256 !== d.attentionSha256)
    errors.push(`${d.file}: attention sha256 differs from manifest`);
  if (art.precomputed !== true) errors.push(`${d.file}: must be labelled precomputed`);
}

const model = json('models/pythia-14m/manifest.json');
check(`models/pythia-14m/${model.weights.file}`, model.weights.sha256);
check(`models/pythia-14m/${model.tokenizer.file}`, model.tokenizer.sha256);
const total = model.weights.tensors.reduce((n, t) => n + t.elementCount * 2, 0);
if (total !== model.weights.byteLength) errors.push('pythia-14m: tensor table does not cover weights.bin');

for (const t of json('tokenizers/manifest.json').tokenizers) check(t.path, t.sha256);

if (errors.length) {
  console.error(`Artifact verification FAILED (${errors.length} problem(s)):`);
  for (const e of errors) console.error('  - ' + e);
  process.exit(1);
}
console.log(
  `Artifact verification passed: ${checked} checksums, ${demos.demos.length} demos, 1 browser model.`,
);
