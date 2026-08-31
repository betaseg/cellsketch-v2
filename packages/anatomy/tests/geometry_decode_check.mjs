// Decodes a geometry payload with the viewer's own decoder, so the binary contract
// between mesh.py and the 3D widgets is covered by the Python suite.
//
//   node geometry_decode_check.mjs <plugin.js> <payload.bin> [--per-index 2]
//   node geometry_decode_check.mjs <plugin.js> <a.bin> --merge <b.bin> [--offset x,y,z]
//
// Prints the vertex/index counts and the bounding box the widget would draw; with
// --merge, the same for the single geometry the two payloads are merged into.
import { readFileSync } from 'node:fs';

const [pluginPath, payloadPath, ...rest] = process.argv.slice(2);
const perIndex = rest.includes('--per-index') ? Number(rest[rest.indexOf('--per-index') + 1]) : 3;
const mergeWith = rest.includes('--merge') ? rest[rest.indexOf('--merge') + 1] : null;
const offset = rest.includes('--offset')
  ? rest[rest.indexOf('--offset') + 1].split(',').map(Number)
  : [0, 0, 0];
// "dist:nx,ny,nz:factor" - the polarity of one instance and the slider's value.
const exploding = rest.includes('--explode') ? rest[rest.indexOf('--explode') + 1] : null;
const plugin = await import(pluginPath);

// Just enough of three.js to hold the arrays mergeInstances builds. The merge is index
// arithmetic over typed arrays; nothing in it needs a GPU, or three itself.
const STUB_THREE = {
  BufferAttribute: class { constructor(array, itemSize) { this.array = array; this.itemSize = itemSize; } },
  BufferGeometry: class {
    setAttribute(name, attribute) { this[name] = attribute; }
    setIndex(attribute) { this.index = attribute; }
    computeVertexNormals() { this.normalsComputed = true; }
  },
};

// Read into a larger buffer at an odd offset: that is how a payload actually arrives -
// a view into an Arrow batch, at whatever offset the column landed on. A decoder that
// assumes alignment throws here rather than in someone's browser.
const raw = readFileSync(payloadPath);
const padded = new Uint8Array(raw.byteLength + 3);
padded.set(raw, 3);
const payload = padded.subarray(3);

let { positions, indices } = plugin.decodePayload(payload, perIndex);
let colours = null;
if (mergeWith) {
  const second = plugin.decodePayload(readFileSync(mergeWith), perIndex);
  const merged = plugin.mergeInstances(STUB_THREE, [
    { geometry: { positions, indices }, rgb: [1, 0, 0] },
    { geometry: second, rgb: [0, 0, 1], offset },
  ]);
  positions = merged.position.array;
  indices = merged.index.array;
  colours = merged.color.array;
}

const axis = (i) => {
  const values = [];
  for (let v = i; v < positions.length; v += 3) values.push(positions[v]);
  return { min: Math.min(...values), max: Math.max(...values) };
};

console.log(JSON.stringify({
  vertices: positions.length / 3,
  indices: indices.length / perIndex,
  maxIndex: indices.reduce((a, b) => Math.max(a, b), 0),
  bbox: { x: axis(0), y: axis(1), z: axis(2) },
  // With --merge: the two instances' colours, to show each kept its own.
  colours: colours ? { first: [...colours.slice(0, 3)], last: [...colours.slice(-3)] } : null,
  // With --explode: where the widget would move that instance.
  explode: exploding ? (() => {
    const [dist, n, factor] = exploding.split(':');
    return plugin.explodeOffset(
      { dist: Number(dist), n: n.split(',').map(Number) }, Number(factor));
  })() : null,
}));
