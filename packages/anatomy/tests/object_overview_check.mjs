// Runs the Objects & Structures cross-indexing from the viewer plugin over a fixture, so
// what the overview claims about a batch is covered by the Python suite.
//
//   node object_overview_check.mjs <plugin.js> <fixture.json>
//
// Fixture: { objects: [{id, membrane}], entityRows: [{object_id, name, kind, instances}] }.
import { readFileSync } from 'node:fs';

const [pluginPath, fixturePath, mode] = process.argv.slice(2);
const plugin = await import(pluginPath);
let { objects, entityRows } = JSON.parse(readFileSync(fixturePath, 'utf8'));

// DuckDB hands back int64 counts as BigInt; a run against a real report sees those where
// a JSON fixture has plain numbers.
if (mode === '--bigint') {
  entityRows = entityRows.map((r) => ({
    ...r, instances: r.instances === null ? null : BigInt(r.instances),
  }));
}

const { entities, valueAt, sizeAt, chosen } = plugin.crossIndex(objects, entityRows);

console.log(JSON.stringify({
  entities: entities.map((e) => `${e.name}:${e.kind}`),
  gaps: plugin.gapsSentence(objects, entities, valueAt),
  mixedDims: plugin.mixedDimsSentence(objects),
  coverage: plugin.objectLevelCoverage(objects),
  // The numbers the tile preview shows before the widget is opened.
  counts: plugin.overviewCounts(objects, entities, entityRows, valueAt),
  short: Object.fromEntries(plugin.shortLabels(objects.map((c) => c.id))),
  matrix: Object.fromEntries(objects.map((object) => [object.id, Object.fromEntries(
    entities.map((entity) => {
      const value = valueAt(object.id, entity.name);
      return [entity.name, value === plugin.MISSING ? 'missing' : value];
    })
  )])),
  sizes: Object.fromEntries(objects.map((object) => [object.id, Object.fromEntries(
    entities.map((entity) => [entity.name, sizeAt(object.id, entity.name)])
  )])),
  // What every swatch, bar and mesh of this structure is drawn in: the report's colour where
  // the run was given one, its place in the palette otherwise.
  colours: Object.fromEntries(
    entities.map((e) => [e.name, plugin.entityColour(e.name, entities.map((x) => x.name), chosen)])),
}));
