// Print every SQL statement the widgets build, so a test can run them against a real report.
//
//   node widget_sql_check.mjs <plugin.js> <group-column> <structure> <target>
//
// The ctx here mirrors the helpers the viewer passes in (viewer/src/renderer.js). Nothing is
// executed: queryRows records the statement and returns nothing, so the builders run without
// a database and without a DOM.

// Enough DOM for the chart helpers, which create elements, set a style and hand them to
// ctx.plot.append. Nothing here renders; the point is to run the code.
const element = () => ({
  style: { cssText: '' },
  appendChild: (child) => child,
  append: () => {},
  replaceChildren: () => {},
  querySelector: () => element(),
  addEventListener: () => {},
  set textContent(_value) {},
  set innerHTML(_value) {},
});
globalThis.document = { createElement: element };
globalThis.window = { location: { search: '' }, addEventListener: () => {} };
globalThis.history = { replaceState: () => {} };

const [pluginPath, groupCol, structure, target] = process.argv.slice(2);
const plugin = await import(pluginPath);

const q = (name) => `"${String(name).replaceAll('"', '""')}"`;
const andWhere = (where, condition) =>
  !condition ? where : (where ? `${where} AND ${condition}` : `WHERE ${condition}`);

const statements = [];
let instanceProfile = null;
const ctx = {
  where: '',
  groups: ['a', 'b'],
  groupLabel: (g) => String(g),
  // Everything the report can hold, so the builders take their fullest branch.
  schema: {
    allCols: ['instance_label', 'contact_count', 'total_volume_um3', 'object_volume_um3',
              'spatial_dims', 'object_mask_name', 'entity_name', 'entity_kind',
              'entity_colour', 'instance_count'],
    metricCols: [], blobCols: [], dimensionInfo: {},
  },
  state: { showSignificance: false },
  color: { group: () => '#000' },
  plot: {
    append: () => {},
    statTable: () => element(),
    prependWarning: () => {},
    // The engine runs the caller's spec the way the viewer does: it queries, so the SQL a
    // chart hands it is recorded and checked like any other statement.
    engine: {
      renderDistribution: async (_container, c, spec) => {
        await c.queryRows(
          `SELECT ${spec.catSql} AS cat, COUNT(${q(spec.numCol)}) AS n `
          + `FROM ${spec.source.table} ${andWhere(spec.source.where, `${q(spec.numCol)} IS NOT NULL`)} `
          + 'GROUP BY 1');
        return true;
      },
    },
  },
  sql: {
    q,
    andWhere,
    groupCol: () => q(groupCol),
    groupExpr: () => `${q(groupCol)} AS __group__`,
  },
  queryRows: async (sql) => {
    statements.push(sql);
    return [];
  },
};

// Every source expression and query the widget file exports. A source is wrapped in a trivial
// SELECT so it is a runnable statement too.
if (plugin.contactEdgeSource) {
  const selected = `(SELECT * FROM ${plugin.contactEdgeSource(ctx)} WHERE ${q('contact_gap_um')} <= 1)`;
  statements.push(`SELECT * FROM ${selected} LIMIT 0`);
  statements.push(plugin.pairCountsSql(ctx, selected));
  instanceProfile = plugin.instanceProfileSql(ctx, { edges: selected, structures: [structure] });
  statements.push(`SELECT * FROM ${instanceProfile} LIMIT 0`);
  statements.push(plugin.perObjectSql(ctx, instanceProfile));
  statements.push(plugin.perObjectGapSql(ctx, selected));
  statements.push(plugin.clusterReachSql(ctx));
  statements.push(`SELECT * FROM ${plugin.valuesTable(
    [{ grp: 'a', value: 1.5 }, { grp: 'b', value: null }],
    [{ key: 'grp', type: 'VARCHAR' }, { key: 'value', type: 'DOUBLE' }])}`);
  statements.push(`SELECT * FROM ${plugin.unnestedSource(ctx, ['instance_entity', 'instance_volume_um3'])} LIMIT 0`);
  statements.push(`SELECT * FROM ${plugin.reachSource(ctx, structure)} LIMIT 0`);
}
if (plugin.objectRowSql) {
  statements.push(plugin.objectRowSql(ctx));
  statements.push(plugin.entityRowSql(ctx, `'an_object'`));
}

// Draw every chart the widget file exports, with rows shaped like the queries return. A
// missing helper or a typo in a plotting path only shows up when the function actually runs.
const drawn = [];
// Counts as DuckDB returns them: BigInt.
const pairRows = [
  { grp: 'a', s1: structure, s2: structure, n: 4n },
  { grp: 'a', s1: structure, s2: target, n: 2n },
  { grp: 'b', s1: structure, s2: target, n: 1n },
];
if (plugin.partnerMix) {
  if (plugin.partnerMix(ctx, element(), { rows: pairRows, structures: [structure, target] })) {
    drawn.push('partnerMix');
  }
  // Every row of the matrix adds up to 100%: shares are per structure, over its own total.
  const shares = plugin.partnerShares(pairRows);
  const rowSums = [...shares.values()].flatMap((structures) => [...structures.values()].map(
    (own) => [...own.partners.values()].reduce((a, b) => a + b, 0) / own.total));
  if (rowSums.length && rowSums.every((sum) => Math.abs(sum - 1) < 1e-9)) drawn.push('shares');
}
if (plugin.reachAgainstChance) {
  // Eight instances of one structure in one object, at 0.1 .. 0.8 µm from the target. Two of
  // them are a group; the other six are on their own.
  const labels = [1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n];
  const groups = plugin.contactGroups(
    labels.map((label) => ({ object_id: 'an_object', entity: structure, label })),
    [{ object_id: 'an_object', entity_a: structure, label_a: 1n,
       entity_b: structure, label_b: 2n }]);
  const rows = labels.map((label, i) => ({
    object_id: 'an_object', entity: structure, label, target, distance: (i + 1) / 10,
  }));
  const records = plugin.clusterReach({
    rows, groups, grpOf: new Map([['an_object', 'a']]),
  });
  // One record: the pair, reaching 0.1 because its closest member does.
  const pair = records.length === 1 ? records[0] : null;
  if (pair && pair.size === 2 && pair.made === structure
      && Math.abs(pair.observed - 0.1) < 1e-9
      && plugin.reachAgainstChance(ctx, element(), { records })) {
    drawn.push('reach');
  }
  // A random pair holds the closest of eight instances a quarter of the time, so a pair that
  // holds it beats the other three quarters, and half of its own quarter.
  const beats = plugin.nullPercentile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 2, 0.1);
  // A group holding every instance has nothing to be compared against.
  if (Math.abs(beats - 87.5) < 1e-9
      && plugin.nullPercentile([0.1, 0.4, 0.6], 3, 0.1) === null) {
    drawn.push('chance');
  }
  // A structure the group is made of is left out: it would be measuring it against itself.
  const ownKind = plugin.clusterReach({
    rows: rows.map((r) => ({ ...r, target: structure })),
    groups,
    grpOf: new Map([['an_object', 'a']]),
  });
  if (!ownKind.length) drawn.push('ownKindLeftOut');
}
if (plugin.objectComparison) {
  const rows = [
    { __cs_group__: 'a', pct_in_contact: 90, mean_partners: 3, largest_cluster: 5, median_gap: 0.1 },
    { __cs_group__: 'b', pct_in_contact: 80, mean_partners: 2, largest_cluster: null, median_gap: 0.2 },
  ];
  const missing = await plugin.objectComparison(ctx, element(), { rows });
  if (Array.isArray(missing) && !missing.length) drawn.push('comparison');
}
if (plugin.clustersByObject) {
  const clusters = plugin.clustersByObject(plugin.contactGroups(
    [{ object_id: 'an_object', entity: structure, label: 1n },
     { object_id: 'an_object', entity: structure, label: 2n },
     { object_id: 'an_object', entity: target, label: 1n }],
    [{ object_id: 'an_object', entity_a: structure, label_a: 1n,
       entity_b: structure, label_b: 2n }]));
  // A chain of two of one structure: a cluster of two, and not a mixed one.
  const entry = clusters.get('an_object');
  if (entry?.largest === 2 && entry.mixed === 0) drawn.push('clusters');
}

console.log(JSON.stringify({ statements, drawn }, null, 1));
