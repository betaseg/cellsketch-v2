// Runs the Contacts & Groups clustering from the viewer plugin over a fixture, so the
// grouping logic is covered by the Python test suite even though it is JavaScript.
//
//   node contact_groups_check.mjs <plugin.js> <fixture.json>
//
// Fixture: { instances: [{object_id, entity, label}], edges: [{object_id, entity_a, label_a,
// entity_b, label_b}] }. Prints the totals and one summary per object.
import { readFileSync } from 'node:fs';

const [pluginPath, fixturePath, mode] = process.argv.slice(2);
const plugin = await import(pluginPath);
let { instances, edges } = JSON.parse(readFileSync(fixturePath, 'utf8'));

// DuckDB returns int64 columns as BigInt, so a run against a real report sees BigInt label
// ids where a JSON fixture has plain numbers. --bigint reproduces that.
if (mode === '--bigint') {
  instances = instances.map((i) => ({ ...i, label: BigInt(i.label) }));
  edges = edges.map((e) => ({ ...e, label_a: BigInt(e.label_a), label_b: BigInt(e.label_b) }));
}

const groups = plugin.contactGroups(instances, edges);

console.log(JSON.stringify({
  total: groups.total,
  singletons: [...groups.sizes.values()].filter((size) => size === 1).length,
  clusters: [...groups.sizes.values()].filter((size) => size > 1).length,
  objects: Object.fromEntries(plugin.clustersByObject(groups)),
}));
