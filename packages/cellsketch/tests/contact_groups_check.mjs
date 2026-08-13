// Runs the Contacts & Groups clustering from the viewer plugin over a fixture, so the
// grouping logic is covered by the Python test suite even though it is JavaScript.
//
//   node contact_groups_check.mjs <plugin.js> <fixture.json>
//
// Fixture: { instances: [{cell_id, entity, label}], edges: [{cell_id, entity_a, label_a,
// entity_b, label_b}], facets: {cell_id: facet} }. Prints one JSON summary per facet.
import { readFileSync } from 'node:fs';

const [pluginPath, fixturePath] = process.argv.slice(2);
const plugin = await import(pluginPath);
const { instances, edges, facets } = JSON.parse(readFileSync(fixturePath, 'utf8'));

const groups = plugin.contactGroups(instances, edges);
const byFacet = plugin.summariseByFacet(groups, new Map(Object.entries(facets)));

console.log(JSON.stringify({
  total: groups.total,
  singletons: [...groups.sizes.values()].filter((size) => size === 1).length,
  clusters: [...groups.sizes.values()].filter((size) => size > 1).length,
  facets: Object.fromEntries([...byFacet].map(([facet, e]) => [facet, {
    instances: e.instances,
    groups: e.sizes.length,
    largest: e.sizes.length ? Math.max(...e.sizes) : 0,
    touching: e.touching,
    sizes: [...e.sizes].sort((a, b) => a - b),
  }])),
}));
