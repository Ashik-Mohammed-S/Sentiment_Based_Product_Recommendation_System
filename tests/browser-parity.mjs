import fs from 'node:fs';
import assert from 'node:assert/strict';
import {recommend} from '../web/engine.mjs';
const data=JSON.parse(fs.readFileSync(new URL('../web/model.json',import.meta.url)));
const fixtures=JSON.parse(fs.readFileSync(process.argv[2]));
for(const f of fixtures){const got=recommend(data,f.liked,f.excluded,f.limit,f.category,f.query);assert.deepEqual(got.map(p=>p.id),f.expected.map(p=>p.id));for(let i=0;i<got.length;i++)assert.ok(Math.abs(got[i].score-f.expected[i].score)<1e-7);}
console.log(`Browser/Python parity: ${fixtures.length} scenarios passed`);
