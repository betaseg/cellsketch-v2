/**
 * Anatomy object overview: what was segmented in each object, before any statistics.
 *
 * Objects are segmented by hand, so one is often missing a structure or has a nucleus split
 * into three labels. A violin drawn across objects hides that; a presence matrix does not.
 *
 * Reads the per-entity rows (obs_level = 1, one per channel), which already carry the count
 * and extent per structure, so it is one query with no unnesting.
 *
 * Self-contained: plugins load as separate ES modules and a static build inlines each as its
 * own data: URL, so a shared-helpers import would not resolve.
 */

const OBJECT_ROW = '"obs_level" = 0';
// One channel of one object. entity_name pins it to this loader's rows rather than another
// per-slice aggregation at the same level.
const ENTITY_ROW = '"obs_level" = 1 AND "entity_name" IS NOT NULL';

const MISSING = Symbol('missing');

const esc = (v) => `'${String(v).replace(/'/g, "''")}'`;

function emptyState(container, text) {
  const p = document.createElement('p');
  p.style.cssText = 'font-size:.85rem;color:#666';
  p.textContent = text;
  container.appendChild(p);
}

/** The question this widget answers, and how to read the answer. */
function caption(container, html) {
  const p = document.createElement('p');
  p.style.cssText = 'font-size:.75rem;color:#666;line-height:1.4;margin:.1rem .2rem .4rem';
  p.innerHTML = html;
  container.appendChild(p);
}

/**
 * Drop the underscore-separated fields every object id has in common, keeping the part that
 * differs; the full id stays as the row's tooltip. Fields, not characters, so a date is not
 * cut in half.
 */
function shortLabels(ids) {
  const parts = ids.map((id) => String(id).split('_'));
  const short = new Map();
  if (ids.length < 2) {
    ids.forEach((id) => short.set(id, String(id)));
    return short;
  }
  const common = (index) => {
    const at = (p) => p[index < 0 ? p.length + index : index];
    const first = at(parts[0]);
    return parts.every((p) => p.length > 1 && at(p) === first);
  };
  let head = 0;
  while (common(head) && head < parts[0].length - 1) head += 1;
  let tail = 0;
  while (common(-1 - tail) && head + tail < parts[0].length - 1) tail += 1;
  ids.forEach((id, i) => {
    const kept = parts[i].slice(head, parts[i].length - tail);
    short.set(id, kept.join('_') || String(id));
  });
  return short;
}

/** Masks before labels, the object mask first of all, then alphabetical. */
function orderEntities(entities, objectMasks) {
  return [...entities].sort((a, b) => {
    const kind = (e) => (e.kind === 'mask' ? 0 : 1);
    const bounds = (e) => (objectMasks.has(e.name) ? 0 : 1);
    return kind(a) - kind(b) || bounds(a) - bounds(b) || a.name.localeCompare(b.name);
  });
}

/**
 * The four numbers the section opens with, drawn on the tile.
 *
 * `uneven` counts structures absent from at least one object. That decides whether the batch
 * can be pooled at all, so it travels with the rest.
 */
function overviewCounts(objects, entities, entityRows, valueAt) {
  return {
    objects: objects.length,
    structures: entities.length,
    labelled: entities.filter((e) => e.kind !== 'mask').length,
    instances: entityRows.reduce((sum, r) => sum + (Number(r.instances) || 0), 0),
    uneven: gapList(objects, entities, valueAt).length,
  };
}

/** The four counts, in the order they answer "how big is this batch, and how even". */
function countEntries(counts) {
  return [
    { value: counts.objects, label: counts.objects === 1 ? 'object' : 'objects' },
    { value: counts.structures, label: 'structures', alert: counts.uneven > 0 },
    { value: counts.labelled, label: 'with instances' },
    { value: counts.instances, label: 'instances' },
  ];
}

/**
 * The tile preview: the four counts as a 2x2 grid rather than a plot, because four scalars
 * make a poor bar chart. An uneven batch marks the structure count in amber; the message band names
 * which structures.
 */
function countGrid(container, counts) {
  const grid = document.createElement('div');
  grid.style.cssText =
    'position:absolute;inset:0;display:grid;grid-template-columns:1fr 1fr;'
    + 'grid-template-rows:1fr 1fr;align-items:center';
  for (const entry of countEntries(counts)) {
    const slot = document.createElement('div');
    slot.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:.1rem';
    const big = document.createElement('div');
    big.style.cssText = 'font-size:1.7rem;font-weight:600;line-height:1.1;'
      + `color:${entry.alert ? '#8a5a00' : '#212529'}`;
    big.textContent = Number(entry.value).toLocaleString();
    const small = document.createElement('div');
    small.style.cssText =
      'font-size:.66rem;color:#666;text-transform:uppercase;letter-spacing:.03em;'
      + 'text-align:center';
    small.textContent = entry.label;
    slot.append(big, small);
    if (entry.alert) {
      const flag = document.createElement('div');
      flag.style.cssText = 'font-size:.62rem;color:#8a6d3b;text-align:center';
      flag.textContent = `⚠ ${counts.uneven} not in every object`;
      slot.appendChild(flag);
    }
    grid.appendChild(slot);
  }
  container.appendChild(grid);
}

// The same order and palette the 3D widgets use, so a structure keeps one colour across the
// report. Duplicated by value because plugins are separate modules with no shared imports.
// Tableau 10 rather than a muted chart palette: the same ten hues in the same order, but
// saturated enough to survive being multiplied by a light and read across a page.
const ENTITY_COLOURS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
];

/**
 * What to draw a structure in: what the report says, else its place in the palette.
 *
 * A run can be given a settings file naming a colour per structure, and it lands on the entity
 * rows. Reading it here rather than keeping a palette per widget is what makes a swatch, a bar
 * and a mesh the same colour.
 */
function entityColour(name, order, chosen) {
  return chosen?.get(name) ?? ENTITY_COLOURS[Math.max(0, order.indexOf(name)) % ENTITY_COLOURS.length];
}

/**
 * How much of an object each structure accounts for, and how much nothing does.
 *
 * Fractions of the object mask's own extent. Structures can overlap, a granule inside the ER
 * counts in both, so the parts can exceed the whole. When they do there is no unassigned
 * remainder to report and the overlap is what the row is telling you.
 */
function composition(objectSize, entities, sizeAt, objectId, objectMask) {
  if (!objectSize) return null;
  const parts = entities
    .filter((e) => e.name !== objectMask)
    .map((e) => ({ name: e.name, fraction: (sizeAt(objectId, e.name) ?? 0) / objectSize }))
    .filter((p) => p.fraction > 0);
  const assigned = parts.reduce((sum, p) => sum + p.fraction, 0);
  return { parts, assigned, unassigned: Math.max(0, 1 - assigned) };
}

/**
 * How much of each object is which structure, as one stacked bar per object.
 *
 * Read across a row for one object; read down a colour for one structure. Percentages of the
 * object mask, so the bars are comparable however different the objects are in size.
 */
function compositionPlot(ctx, container, { objects, entities, sizeAt, shortOf, size, chosen }) {
  const order = entities.map((e) => e.name);
  const shares = objects.map((object) => composition(
    object.size, entities, sizeAt, object.id, object.objectMask));
  if (!shares.some(Boolean)) return false;

  const labels = objects.map((o) => shortOf.get(o.id) ?? o.id);
  const inside = entities.filter((e) => e.name !== objects[0]?.objectMask);
  const traces = inside.map((entity) => ({
    type: 'bar',
    orientation: 'h',
    name: entity.name,
    y: labels,
    x: shares.map((share) => (share?.parts.find((p) => p.name === entity.name)?.fraction ?? 0) * 100),
    marker: { color: entityColour(entity.name, order, chosen) },
    hovertemplate: `%{y}<br>${entity.name}: %{x:.1f}% of the object<extra></extra>`,
  }));
  traces.push({
    type: 'bar',
    orientation: 'h',
    name: 'in no structure',
    y: labels,
    x: shares.map((share) => (share?.unassigned ?? 0) * 100),
    marker: { color: '#dcdcdc' },
    hovertemplate: '%{y}<br>in no structure: %{x:.1f}%<extra></extra>',
  });

  ctx.plot.append(container, traces, {
    title: { text: `Composition: share of each object's ${size.noun}` },
    barmode: 'stack',
    xaxis: { title: '% of the object', rangemode: 'tozero' },
    yaxis: { automargin: true, autorange: 'reversed' },
    showlegend: true,
    legend: { orientation: 'h', y: -0.2 },
    height: Math.max(220, 34 * objects.length + 120),
  });
  return true;
}

function dot(color) {
  const span = document.createElement('span');
  span.style.cssText =
    `display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};`
    + 'margin-right:5px;flex-shrink:0';
  return span;
}

/**
 * The presence matrix: one row per object, one column per structure. DOM rather than
 * statTable, because a missing structure has to be visible without reading the numbers.
 */
function presenceTable(ctx, { objects, entities, valueAt, shortOf, hasGroups, size, chosen }) {
  const table = document.createElement('table');
  table.className = 'stat-table';
  table.style.cssText = 'font-size:.78rem';
  const order = entities.map((e) => e.name);

  const head = document.createElement('tr');
  const columns = [...(hasGroups ? ['group'] : []), 'object',
                   `${size.noun} (${size.unit})`];
  for (const title of columns) {
    head.appendChild(Object.assign(document.createElement('th'), { textContent: title }));
  }
  for (const entity of entities) {
    const th = document.createElement('th');
    th.style.textAlign = 'center';
    // The dot ties the column to its segment in the composition bar.
    th.append(dot(entityColour(entity.name, order, chosen)), entity.name);
    th.appendChild(document.createElement('br'));
    const kind = document.createElement('span');
    kind.style.cssText = 'font-size:.68rem;font-weight:400;color:#888';
    kind.textContent = entity.kind;
    th.appendChild(kind);
    head.appendChild(th);
  }
  const thead = document.createElement('thead');
  thead.appendChild(head);

  const tbody = document.createElement('tbody');
  for (const object of objects) {
    const tr = document.createElement('tr');
    if (hasGroups) {
      const td = document.createElement('td');
      td.style.whiteSpace = 'nowrap';
      td.append(dot(ctx.color.group(object.group)), ctx.groupLabel(object.group));
      tr.appendChild(td);
    }
    const name = document.createElement('td');
    name.textContent = shortOf.get(object.id) ?? object.id;
    name.title = object.id;
    tr.appendChild(name);

    // The object's own extent, so a bigger object is visible as such.
    const total = document.createElement('td');
    total.style.textAlign = 'right';
    total.textContent = object.size === null ? '-' : Number(object.size).toPrecision(4);
    total.title = object.size === null
      ? `no ${object.objectMask ?? 'object mask'} ${size.noun} for ${object.id}`
      : `${object.id}: ${object.size} ${size.unit}`;
    tr.appendChild(total);

    for (const entity of entities) {
      const td = document.createElement('td');
      td.style.textAlign = 'center';
      const value = valueAt(object.id, entity.name);
      if (value === MISSING) {
        td.textContent = 'missing';
        td.style.cssText = 'text-align:center;color:#b3261e;background:#fdecea;font-weight:500';
        td.title = `${entity.name} was not segmented for ${object.id}`;
      } else if (value === null) {
        // A mask has no instances to count: it is one structure, present or not.
        td.textContent = '✓';
        td.style.color = '#2e7d32';
      } else {
        td.textContent = Number(value).toLocaleString();
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  table.append(thead, tbody);
  const scroll = document.createElement('div');
  scroll.style.cssText = 'overflow-x:auto;max-width:100%';
  scroll.appendChild(table);
  return scroll;
}

/** Structures that are not in every object, and how many objects each is missing from. */
function gapList(objects, entities, valueAt) {
  return entities
    .map((e) => ({
      name: e.name,
      missing: objects.filter((c) => valueAt(c.id, e.name) === MISSING).length,
    }))
    .filter((e) => e.missing > 0);
}

/**
 * Whether this report pools planes and volumes, as a sentence. An area and a volume do not
 * belong in one distribution, so this is the same class of warning as a missing structure.
 */
function mixedDimsSentence(objects) {
  const counted = new Map();
  for (const object of objects) {
    if (object.dims === null) continue;
    counted.set(object.dims, (counted.get(object.dims) ?? 0) + 1);
  }
  if (counted.size < 2) return null;
  const listed = [...counted.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([dims, n]) => `<b>${n}</b> in ${dims}D`)
    .join(', ');
  return `This report mixes objects of different dimensionality: ${listed}. An area and a `
    + 'volume are not comparable, so the size charts below cover only one of them and any '
    + 'statistic pooling both means nothing.';
}

/**
 * Which object-level measurements are missing, and from how many objects.
 *
 * Both come from a processor that sees the whole object, so an object can carry its
 * per-structure rows and still have neither: excluded with --no-instances / --no-contacts,
 * or the object failed. Anything pooled over objects is then pooled over fewer than it
 * appears.
 */
function objectLevelCoverage(objects) {
  return [
    { label: 'per-instance measurements', present: objects.filter((o) => o.hasInstances).length },
    { label: 'contacts', present: objects.filter((o) => o.hasContacts).length },
  ];
}

/** Which structures are not in every object, as a sentence. */
function gapsSentence(objects, entities, valueAt) {
  const gaps = gapList(objects, entities, valueAt);
  if (!gaps.length) return null;
  const listed = gaps
    .map((g) => `<b>${g.name}</b> in ${g.missing} of ${objects.length}`)
    .join(', ');
  return `Not every object was segmented the same way: ${listed} missing.`;
}

/**
 * What an object's extent is called here: volume for a 3D batch, area for a 2D one. A report
 * holds one or the other, so the rest of the widget speaks of "size".
 */
function sizeVocabulary(schema) {
  const planar = !schema.allCols.includes('total_volume_um3')
    && schema.allCols.includes('total_area_um2');
  return planar
    ? { object: 'object_area_um2', total: 'total_area_um2', noun: 'area', unit: 'µm²' }
    : { object: 'object_volume_um3', total: 'total_volume_um3', noun: 'volume', unit: 'µm³' };
}

function objectRowSql(ctx) {
  const size = sizeVocabulary(ctx.schema);
  const has = (column) => (ctx.schema.allCols.includes(column)
    ? `${ctx.sql.q(column)} IS NOT NULL` : 'FALSE');
  return `SELECT ${ctx.sql.q('object_id')} AS id,
            ${ctx.sql.groupCol()} AS grp,
            ${ctx.sql.q('object_mask_name')} AS object_mask,
            ${ctx.sql.q('spatial_dims')} AS dims,
            ${ctx.sql.q(size.object)} AS size,
            ${has('instance_label')} AS has_instances,
            ${has('contact_count')} AS has_contacts
     FROM pp_data ${ctx.sql.andWhere(ctx.where, OBJECT_ROW)}`;
}


function entityRowSql(ctx, ids) {
  const size = sizeVocabulary(ctx.schema);
  return `SELECT ${ctx.sql.q('object_id')} AS object_id,
            ${ctx.sql.q('entity_name')} AS name,
            ${ctx.sql.q('entity_kind')} AS kind,
            ${ctx.schema.allCols.includes('entity_colour')
              ? ctx.sql.q('entity_colour') : 'NULL'} AS colour,
            ${ctx.sql.q('instance_count')} AS instances,
            ${ctx.sql.q(size.total)} AS size
     FROM pp_all WHERE ${ENTITY_ROW} AND ${ctx.sql.q('object_id')} IN (${ids})`;
}


async function readObjects(ctx) {
  const objectRows = await ctx.queryRows(objectRowSql(ctx));
  const objects = objectRows
    .filter((r) => r.id !== null && r.id !== undefined)
    .map((r) => ({
      id: String(r.id),
      group: r.grp,
      objectMask: r.object_mask,
      dims: r.dims === null || r.dims === undefined ? null : Number(r.dims),
      size: r.size === null || r.size === undefined ? null : Number(r.size),
      hasInstances: Boolean(r.has_instances),
      hasContacts: Boolean(r.has_contacts),
    }));
  if (!objects.length) return { objects: [], entityRows: [] };

  // pp_all: the per-entity rows sit a level below the object rows in pp_data. Restricted to
  // the objects the filter left, since a filter on object columns cannot be applied to a row
  // that has none of them.
  const ids = objects.map((c) => esc(c.id)).join(', ');
  const entityRows = await ctx.queryRows(entityRowSql(ctx, ids));
  return { objects, entityRows };
}

/** Objects and entities, cross-indexed, with the lookups the table and charts need. */
function crossIndex(objects, entityRows) {
  const byKey = new Map();
  const entities = new Map();
  for (const row of entityRows) {
    const name = row.name === null || row.name === undefined ? null : String(row.name);
    if (!name) continue;
    entities.set(name, {
      name,
      kind: String(row.kind ?? 'label'),
      colour: row.colour === null || row.colour === undefined ? null : String(row.colour),
    });
    byKey.set(`${row.object_id}|${name}`, {
      instances: row.instances === null || row.instances === undefined
        ? null : Number(row.instances),
      size: row.size === null || row.size === undefined ? null : Number(row.size),
    });
  }
  const objectMasks = new Set(objects.map((o) => o.objectMask).filter(Boolean).map(String));
  const ordered = orderEntities(entities.values(), objectMasks);
  const valueAt = (objectId, name) => {
    const found = byKey.get(`${objectId}|${name}`);
    return found ? found.instances : MISSING;
  };
  const sizeAt = (objectId, name) => byKey.get(`${objectId}|${name}`)?.size ?? null;
  // What the run was told to draw each structure in, where it was told anything.
  const chosen = new Map([...entities.values()]
    .filter((e) => e.colour).map((e) => [e.name, e.colour]));
  return { entities: ordered, valueAt, sizeAt, chosen };
}

function sortObjects(objects, ctx) {
  const groupRank = new Map(ctx.groups.map((g, i) => [String(g), i]));
  return [...objects].sort((a, b) => {
    const ra = groupRank.get(String(a.group)) ?? Number.MAX_SAFE_INTEGER;
    const rb = groupRank.get(String(b.group)) ?? Number.MAX_SAFE_INTEGER;
    return ra - rb || a.id.localeCompare(b.id);
  });
}

const objectOverview = {
  id: 'anatomy-objects',
  label: 'Objects & Structures',
  group: 'Summary',
  // The viewer renders this behind the ⓘ in the card header, and the sidebar's info switch
  // opens every one of them at once.
  info: [
    'What was **actually segmented**, object by object. Read this before any distribution.',
    '',
    'A number is how many instances of that structure the object has, **✓** is a '
    + 'whole-structure mask, and **missing** means the object has none of that structure.',
    '',
    'Objects are not segmented identically, and a structure missing from an object silently '
    + 'changes every statistic that pools objects together.',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.includes('object_id') && schema.allCols.includes('entity_name');
  },

  // The counts themselves are the preview; the band below them says whether they can be
  // trusted to pool, and names the structures that cannot.
  async overviewMessage(ctx) {
    const { objects, entityRows } = await readObjects(ctx);
    if (!objects.length) return null;
    const { entities, valueAt } = crossIndex(objects, entityRows);
    const mixed = mixedDimsSentence(objects);
    if (mixed) return { text: mixed, warning: true };
    const gaps = gapsSentence(objects, entities, valueAt);
    if (gaps) return { text: gaps, warning: true };
    const short = objectLevelCoverage(objects).filter((c) => c.present < objects.length);
    if (short.length) {
      return {
        text: `No ${short.map((c) => c.label).join(' or ')} for `
          + `<strong>${objects.length - Math.max(...short.map((c) => c.present))}</strong> `
          + `of <strong>${objects.length}</strong> object(s).`,
        warning: true,
      };
    }
    return `Every object was segmented the same way, into the same `
      + `<strong>${entities.length}</strong> structure(s), so nothing is missing from a `
      + 'statistic that pools them.';
  },

  async overviewPlot(container, ctx) {
    const { objects, entityRows } = await readObjects(ctx);
    if (!objects.length) return false;
    const { entities, valueAt } = crossIndex(objects, entityRows);
    if (!entities.length) return false;
    countGrid(container, overviewCounts(objects, entities, entityRows, valueAt));
    return true;
  },

  async render(container, ctx) {
    const { objects: unsorted, entityRows } = await readObjects(ctx);
    if (!unsorted.length) {
      emptyState(container, 'No objects in this report.');
      return;
    }
    const objects = sortObjects(unsorted, ctx);
    const { entities, valueAt, sizeAt, chosen } = crossIndex(objects, entityRows);
    if (!entities.length) {
      emptyState(container, 'No segmented structures in this report.');
      return;
    }
    const shortOf = shortLabels(objects.map((c) => c.id));
    const hasGroups = ctx.groups.length > 1;

    const gaps = gapsSentence(objects, entities, valueAt);
    if (gaps) {
      ctx.plot.prependWarning(container, { html: gaps });
    }
    const mixed = mixedDimsSentence(objects);
    if (mixed) {
      ctx.plot.prependWarning(container, { html: mixed, level: 'red' });
    }
    ctx.plot.dataAvailabilityWarning(container, objectLevelCoverage(objects), objects.length,
                                     { unit: 'objects', level: 'yellow' });

    const size = sizeVocabulary(ctx.schema);
    container.appendChild(presenceTable(ctx, {
      objects, entities, valueAt, shortOf, hasGroups, size, chosen,
    }));
    caption(container,
      `The ${size.noun} column is the object mask's own, so objects are comparable by it. `
      + 'Counts come from the per-structure rows, so they agree with every other widget by '
      + 'construction.');

    const composed = document.createElement('div');
    composed.style.cssText = 'margin-top:1rem';
    container.appendChild(composed);
    if (compositionPlot(ctx, composed, { objects, entities, sizeAt, shortOf, size, chosen })) {
      caption(composed,
        'What each object is made of, as a share of its own extent. Structures can overlap, a '
        + 'granule inside the ER counts in both, so a row can pass 100%. What is in no '
        + 'structure at all is the grey remainder.');
    }

  },
};

export default [objectOverview];

// Named for the test suite: what the table says about an object is decided here, not in the
// DOM it ends up in.
export { crossIndex, entityColour, gapsSentence, mixedDimsSentence, objectLevelCoverage,
  overviewCounts, shortLabels, MISSING };

// For the SQL harness: the statements this widget builds, so a test can run them.
export { objectRowSql, entityRowSql };
