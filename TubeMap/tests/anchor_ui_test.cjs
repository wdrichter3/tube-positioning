// Exercise the real dashboard handler with controlled asynchronous API responses.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../tubemap/static/app.js'), 'utf8');
const start = source.indexOf('async function saveAnchors(');
const end = source.indexOf('\n$("#anc-add").onclick', start);
assert(start >= 0 && end > start);

function harness() {
  const current = {capture_id: 'capture-a', map_revision: 'revision-a', anchors: []};
  const S = {map: {cid: 'capture-a', data: current, sel: 'T1', generation: 0, loading: false, writing: false}};
  const fields = {'#anc-scale': {checked: false}, '#map-err': {textContent: ''}};
  const requests = [];
  const context = vm.createContext({S, $: id => fields[id] ||= {}, $$: () => [],
    api: (url, options) => new Promise((resolve,reject) => requests.push({url, options, resolve, reject})),
    renderMapResult() {}, mapV: {draw() {}, clear() {}, async load() {}}, mapRoi: {cancel() {}},
    geometryForm: () => ({mode:'known'}), setGeometryForm() {}, fmt: String, toast() {}});
  vm.runInContext(source.slice(source.indexOf('async function mapLoadCapture('), source.indexOf('function renderMapResult(')), context);
  vm.runInContext(source.slice(source.indexOf('async function calibrateDepth('), source.indexOf('function companionHtml(')), context);
  vm.runInContext(source.slice(start, end), context);
  return {S, current, fields, requests, resolve: x => requests[0].resolve(x), reject: x => requests[0].reject(x),
    save: context.saveAnchors, load: context.mapLoadCapture, build: context.buildMap, calibrate: context.calibrateDepth,
    request: () => requests[0]};
}

test('current-map save sends revision and adopts successful response', async () => {
  const h = harness(), pending = h.save([]);
  assert.equal(h.request().url, '/api/captures/capture-a/anchors');
  assert.equal(h.request().options.json.map_revision, 'revision-a');
  const updated = {capture_id: 'capture-a', map_revision: 'revision-next', anchors: [{id: 'T1'}]};
  h.resolve(updated); await pending;
  assert.equal(h.S.map.data, updated);
  assert.equal(h.S.map.sel, 'T1');
});

test('rapid saves are serialized and a subsequent save uses authoritative response revision', async () => {
  const h = harness(), first = h.save([]);
  await h.save([{id:'T2'}]);
  assert.equal(h.requests.length, 1);
  h.resolve({...h.current, map_revision:'next', anchors:[{id:'T1'}]}); await first;
  const next = h.save(h.S.map.data.anchors);
  assert.equal(h.requests[1].options.json.map_revision, 'next');
  h.requests[1].resolve({...h.S.map.data, map_revision:'last'}); await next;
  assert.equal(h.S.map.data.map_revision, 'last');
});

test('capture mismatch never sends anchors to a different capture URL', async () => {
  const h = harness(); h.current.capture_id = 'other';
  await h.save([]);
  assert.equal(h.requests.length, 0);
});

const capture = (cid, revision, roi = null) => ({meta:{image:{width:100,height:100}},
  map:{capture_id:cid, map_revision:revision, roi}, labels:{roi:null}});

test('late preview image cannot overwrite the active capture image', async () => {
  const images = [];
  class FakeImage { constructor() { images.push(this); } }
  const context = vm.createContext({Image:FakeImage});
  const method = source.slice(source.indexOf('  async load(url,'), source.indexOf('  clear() {'));
  vm.runInContext(`class TestViewer { ${method} fit() {} }; var viewer = new TestViewer();`, context);
  let generation = 1;
  const old = context.viewer.load('/old', 100, 100, () => generation === 1);
  generation = 2;
  const fresh = context.viewer.load('/fresh', 200, 200, () => generation === 2);
  images[1].onload(); await fresh;
  images[0].onload(); await old;
  assert.equal(context.viewer.img, images[1]);
  assert.equal(context.viewer.iw, 200);
});

test('out-of-order A/B/A loads clear old ROI and keep only the latest A response', async () => {
  const h = harness(); h.S.map.roi = [[.1,.1]];
  const a1 = h.load('capture-a'), b = h.load('capture-b'), a2 = h.load('capture-a');
  assert.equal(h.S.map.data, null); assert.equal(h.S.map.roi, null); assert.equal(h.S.map.sel, null);
  await h.build(); assert.equal(h.requests.length, 3);
  h.requests[2].resolve(capture('capture-a','latest', [[0,0]])); await a2;
  h.requests[1].resolve(capture('capture-b','b')); await b;
  h.requests[0].resolve(capture('capture-a','old')); await a1;
  assert.equal(h.S.map.data.map_revision, 'latest');
  assert.deepEqual(h.S.map.roi, [[0,0]]);
});

test('late build response cannot replace a newly loaded capture', async () => {
  const h = harness(), build = h.build();
  assert.equal(h.S.map.sel, null);
  const load = h.load('capture-b');
  h.requests[1].resolve(capture('capture-b','b')); await load;
  h.requests[0].resolve({...h.current, map_revision:'late'}); await build;
  assert.equal(h.S.map.data.capture_id, 'capture-b');
});

test('rebuild resets reused selected ID; safe anchor save retains it', async () => {
  const h = harness(), pending = h.build();
  h.resolve({...h.current, map_revision:'rebuilt'}); await pending;
  assert.equal(h.S.map.sel, null);
});

test('depth calibration carries revision and ignores late response after capture switch', async () => {
  const h = harness(), pending = h.calibrate(h.current, 'main');
  assert.equal(h.request().options.json.map_revision, 'revision-a');
  assert.equal(h.S.map.sel, null);
  const load = h.load('capture-b');
  h.requests[1].resolve(capture('capture-b','b')); await load;
  h.requests[0].resolve({map:{...h.current,map_revision:'late'},factor:1.05}); await pending;
  assert.equal(h.S.map.data.capture_id, 'capture-b');
});

test('409 clears stale machine-map display and selection with persistent explanation', async () => {
  const h = harness(), pending = h.save([]);
  h.reject(Object.assign(new Error('Map changed'), {status: 409}));
  await pending;
  assert.equal(h.S.map.data, null);
  assert.equal(h.S.map.sel, null);
  assert.equal(h.fields['#map-err'].textContent, 'Map changed');
});

test('late response cannot replace another capture or a rebuilt map', async () => {
  for (const switchCapture of [false, true]) {
    const h = harness(), pending = h.save([]);
    const newer = {map_revision: 'revision-b'};
    h.S.map.data = newer;
    if (switchCapture) h.S.map.cid = 'capture-b';
    h.resolve({map_revision: 'revision-a'}); await pending;
    assert.equal(h.S.map.data, newer);
  }
});

test('late rejection cannot clear a newer map', async () => {
  const h = harness(), pending = h.save([]);
  const newer = {map_revision: 'revision-b'};
  h.S.map.data = newer;
  h.reject(Object.assign(new Error('Map changed'), {status: 409}));
  await pending;
  assert.equal(h.S.map.data, newer);
  assert.equal(h.fields['#map-err'].textContent, '');
});
