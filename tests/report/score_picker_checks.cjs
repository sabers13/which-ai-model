/* Offline checks of the browser selector against committed measurements. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync('web/assets/page.js', 'utf8'), context);
const choose = context.selectScorePick;
const tasks = JSON.parse(fs.readFileSync('results/page/page-data.json', 'utf8')).tasks;
const expected = {
  'T-ROUTE': {
    65: ['glm-5.3-flash', 'gemini-4-argon', 'deepseek-v4.1-flash'],
    75: ['glm-5.3-flash', 'gemini-4-argon', 'deepseek-v4.1-flash'],
    85: ['gemini-4-argon', 'gemini-4-argon', 'gemini-4-argon']
  },
  'T-SCOPE': {
    65: ['gpt-6-luna', 'claude-haiku-5-5', 'deepseek-v4.1-flash'],
    75: [null, null, null], 85: [null, null, null]
  },
  'T-REPLY': {
    65: ['claude-haiku-5-5', 'gemini-4-argon', 'claude-haiku-5-5'],
    75: ['gemini-4-argon', 'gemini-4-argon', 'gemini-4-argon'],
    85: [null, null, null]
  },
  'T-DOCS': {
    65: ['gpt-6-luna', 'gpt-6-luna', 'claude-haiku-5-5'],
    75: ['gpt-6-luna', 'gpt-6-luna', 'claude-haiku-5-5'],
    85: ['gpt-6-luna', 'gpt-6-luna', 'claude-haiku-5-5']
  }
};
for (const task of tasks) for (const limit of [65, 75, 85]) {
  const original = JSON.stringify(task.rows);
  for (const [index, priority] of ['cheapest', 'most_accurate', 'fastest'].entries()) {
    const result = choose(task.rows, limit / 100, priority);
    assert.equal(result.candidate?.model ?? null, expected[task.id][limit][index], `${task.id} ${limit} ${priority}`);
    assert.equal(result.qualifying, task.rows.filter(row => row.metric.value >= limit / 100).length);
  }
  assert.equal(JSON.stringify(task.rows), original, 'selection must not mutate evidence');
}
const row = (model, score, cost, latency) => ({model, metric:{value:score}, cost:{value:cost}, latency:{value:latency}});
assert.equal(choose([row('exact', .75, 0, 10), row('below', .7499, 0, 1)], .75, 'fastest').candidate.model, 'exact');
assert.equal(choose([row('unknown', .9, null, null), row('known', .8, 1, 20)], .75, 'cheapest').candidate.model, 'known');
assert.equal(choose([row('unknown', .9, null, null)], .75, 'cheapest').reason, 'cost');
assert.equal(choose([row('unknown', .9, null, null)], .75, 'fastest').reason, 'latency');
assert.equal(choose([row('unknown', .9, null, null)], .75, 'most_accurate').candidate.model, 'unknown');
assert.equal(choose([row('b', .9, 2, 10), row('a', .9, 1, 10)], .85, 'most_accurate').candidate.model, 'a');
assert.equal(choose([], .75, 'cheapest').candidate, null);
assert.equal(choose([row('invalid', null, 0, 0)], .65, 'cheapest').qualifying, 0);
assert.equal(choose([row('valid', .9, 0, 1)], NaN, 'cheapest').candidate, null);
console.log('Score picker: all committed combinations, boundaries, ties and unknown values passed');
