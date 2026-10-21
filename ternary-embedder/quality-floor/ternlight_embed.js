// Ternlight vectors for run_mteb.py (embed-pipeline plan, step 0b), 2026-10-16.
//   node ternlight_embed.js <base|mini> <texts.json> <out.npy>
const fs = require('fs');
const path = require('path');

const [tier, inFile, outFile] = process.argv.slice(2);
const mod = require(path.join(__dirname, 'node_modules', '@ternlight', tier));
const texts = JSON.parse(fs.readFileSync(inFile, 'utf8'));
const rows = texts.map((t) => mod.embed(t));
const dim = rows[0].length;

const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${rows.length}, ${dim}), }`;
let h = header;
while ((10 + h.length + 1) % 64 !== 0) h += ' ';
h += '\n';
const pre = Buffer.alloc(10);
pre.write('\x93NUMPY', 0, 'latin1');
pre[6] = 1; pre[7] = 0;
pre.writeUInt16LE(h.length, 8);
const body = Buffer.alloc(rows.length * dim * 4);
rows.forEach((v, i) => { for (let j = 0; j < dim; j++) body.writeFloatLE(v[j], (i * dim + j) * 4); });
fs.writeFileSync(outFile, Buffer.concat([pre, Buffer.from(h, 'latin1'), body]));
