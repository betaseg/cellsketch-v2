/**
 * Anatomy viewer widgets.
 *
 * Per-instance data lives as parallel list columns on the object row (see the package
 * README), so each widget builds a source that unnests those lists into one row per
 * instance and hands it to the viewer's distribution engine. The engine takes any source
 * table expression, so instance-level data gets the same violin/box, palette, grouping and
 * Mann-Whitney significance machinery as the built-in widgets.
 *
 * Exported as an array: one file, four widgets, no cross-file imports to serve.
 */

// Every metric an instance can carry, in either dimensionality. Filtering this by what the
// schema holds is all the branching a widget needs, since a report has one set or the other.
const METRIC_LABELS = {
  instance_volume_um3: 'Volume (µm³)',
  instance_area_um2: 'Area (µm²)',
  instance_surface_area_um2: 'Surface area (µm²)',
  instance_perimeter_um: 'Perimeter (µm)',
  instance_sphericity: 'Sphericity',
  instance_circularity: 'Circularity',
  instance_aspect_ratio_major_minor: 'Aspect ratio (major/minor)',
  instance_branches: 'Skeleton branches',
  instance_length_um: 'Skeleton length (µm)',
  instance_tortuosity: 'Tortuosity',
  instance_distance_to_closest_same_type_um: 'Distance to nearest same type (µm)',
  instance_polar_dist_um: 'Distance from object centre (µm)',
  instance_polar_az_deg: 'Azimuth from object centre (°)',
  instance_polar_el_deg: 'Elevation from object centre (°)',
  instance_polar_angle_deg: 'Angle from object centre (°)',
};

// Size and shape first, then skeleton, then position. Each 3D metric is followed by its 2D
// counterpart, so "the first one this report has" is a size either way.
const METRIC_ORDER = Object.keys(METRIC_LABELS);

const OBJECT_ROW = '"obs_level" = 0';
// Columns carrying text rather than a measurement, so the finite guard below skips them.
const TEXT_COLUMNS = new Set([
  'instance_entity', 'distance_entity', 'distance_target', 'distance_hist_counts',
  'contact_entity_a', 'contact_entity_b',
]);
// The grouping column, aliased once so a source that unnests lists can still carry it.
const GROUP_KEY = '__cs_group__';
const GROUP_ALIAS = `"${GROUP_KEY}"`;

const esc = (v) => `'${String(v).replace(/'/g, "''")}'`;
const labelFor = (col) => METRIC_LABELS[col] ?? col;

/** Distinct values of a scalar SQL expression over the object rows, in sorted order. */
async function distinctValues(ctx, expr) {
  const where = ctx.sql.andWhere(ctx.where, OBJECT_ROW);
  const rows = await ctx.queryRows(
    `SELECT DISTINCT ${expr} AS v FROM pp_data ${where} ORDER BY v`
  );
  return rows.map((r) => r.v).filter((v) => v !== null && v !== undefined);
}

// ── the structure in focus ─────────────────────────────────────────────────────
//
// One choice shared by every Anatomy widget and kept in the URL, so a link opens on the
// structure you were reading about. The viewer's own URL writer only sets and deletes keys it
// knows, so a key of ours survives it. Plugins are separate modules with no shared imports,
// so the value lives in the URL and changes are announced on `window`.

const FOCUS_PARAM = 'structure';
const FOCUS_EVENT = 'anatomy:structure';

/** The focused structure if this report has it, else the first available. */
function focusedStructure(available) {
  const wanted = new URLSearchParams(window.location.search).get(FOCUS_PARAM);
  return available.includes(wanted) ? wanted : available[0];
}

/** Focus a structure: into the URL, then out to the other widgets. */
function focusStructure(name) {
  const params = new URLSearchParams(window.location.search);
  params.set(FOCUS_PARAM, name);
  history.replaceState(null, '', `?${params}`);
  window.dispatchEvent(new CustomEvent(FOCUS_EVENT, { detail: name }));
}

/**
 * Run `handler` when another widget changes the focus, until `container` leaves the page.
 * Widgets are re-rendered by replacing their container, which is what unsubscribes.
 */
function onStructureFocus(container, handler) {
  const listener = (event) => {
    if (!container.isConnected) {
      window.removeEventListener(FOCUS_EVENT, listener);
      return;
    }
    handler(event.detail);
  };
  window.addEventListener(FOCUS_EVENT, listener);
}


/**
 * How many instances of this structure each metric actually measured.
 *
 * A metric can be null for every instance of one structure and fine for the next: a skeleton
 * is skipped above --max-skeleton-voxels, and the nearest same-structure instance needs a
 * second one in the same object. Plotting those leaves holes in the grid, so they are named
 * in a caption instead.
 */
async function measuredMetrics(ctx, metrics, source, where) {
  const counts = metrics
    .map((m) => `COUNT(${ctx.sql.q(m)}) AS ${ctx.sql.q(m)}`)
    .join(', ');
  const [row] = await ctx.queryRows(`SELECT ${counts} FROM ${source} ${where}`);
  return Object.fromEntries(metrics.map((m) => [m, Number(row?.[m] ?? 0)]));
}

/**
 * A source table expression yielding one row per unnested element.
 * Every listed column is unnested in the same SELECT, which keeps them row-aligned.
 */
function unnestedSource(ctx, columns) {
  const selects = columns.map((c) => `unnest(${ctx.sql.q(c)}) AS ${ctx.sql.q(c)}`);
  const where = ctx.sql.andWhere(ctx.where, OBJECT_ROW);
  // object_id rides along: an instance is identified by object + entity + label, which is
  // what any join on instances has to match.
  const inner = `(SELECT "object_id", ${ctx.sql.groupCol()} AS ${GROUP_ALIAS}, ${selects.join(', ')}
                  FROM pp_data ${where})`;
  // A NaN makes DuckDB's STDDEV raise "out of range" and turns quantiles into nan. The
  // processors write NULL instead, but older reports still hold NaNs.
  const guarded = columns.map((c) => {
    const q = ctx.sql.q(c);
    return TEXT_COLUMNS.has(c) ? q : `CASE WHEN isfinite(${q}) THEN ${q} END AS ${q}`;
  });
  return `(SELECT "object_id", ${GROUP_ALIAS}, ${guarded.join(', ')} FROM ${inner})`;
}

/** A labelled <select>; calls onChange with the new value. */
function selector(labelText, values, current, onChange) {
  const wrap = document.createElement('label');
  wrap.style.cssText = 'display:inline-flex;align-items:center;gap:.4rem;font-size:.8rem;margin-right:1rem';
  wrap.textContent = labelText;
  const select = document.createElement('select');
  select.style.cssText = 'font-size:.8rem;padding:.15rem .3rem';
  for (const v of values) {
    const opt = document.createElement('option');
    opt.value = String(v);
    opt.textContent = String(v);
    if (String(v) === String(current)) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener('change', () => onChange(select.value));
  wrap.appendChild(select);
  return wrap;
}

function note(container, text) {
  const p = document.createElement('p');
  p.style.cssText = 'font-size:.78rem;color:#666;margin:.2rem 0 .6rem';
  p.textContent = text;
  container.appendChild(p);
}

/** A block heading: the question the panels under it answer. */
function heading(container, text) {
  const h = document.createElement('h4');
  h.style.cssText = 'font-size:.88rem;font-weight:600;margin:1.2rem 0 .1rem;color:#222';
  h.textContent = text;
  container.appendChild(h);
}

/** A caption under one chart, for what that panel in particular shows. */
function caption(container, html) {
  const p = document.createElement('p');
  p.style.cssText = 'font-size:.75rem;color:#666;line-height:1.4;margin:.1rem .2rem .4rem';
  p.innerHTML = html;
  container.appendChild(p);
}

function emptyState(container, text) {
  const p = document.createElement('p');
  p.style.cssText = 'font-size:.85rem;color:#666';
  p.textContent = text;
  container.appendChild(p);
}

/** One violin per group for `numCol`, over `source`, with significance if enabled. */
/**
 * One value per group is not a distribution.
 *
 * Grouped per object, a structure with one instance each gives every group a single point:
 * a violin of one, and a ladder of Mann-Whitney brackets that can only ever read "ns". Bars
 * and no test say the same thing without pretending.
 */
async function isOnePerCategory(ctx, numCol, source, where, catSql) {
  const [row] = await ctx.queryRows(
    `SELECT MAX(n) AS most FROM (
       SELECT ${catSql} AS c, COUNT(${ctx.sql.q(numCol)}) AS n
       FROM ${source} ${where ?? ''} GROUP BY 1)`
  );
  return Number(row?.most ?? 0) <= 1;
}

/**
 * One violin or box per category for `numCol`, with significance if enabled.
 *
 * The category is the grouping unless a caller passes its own expression, which is how a
 * chart can compare something else (buckets of a column, say) and still get the engine's
 * brackets: the engine tests whatever sits on the X axis.
 */
async function drawDistribution(ctx, slot, {
  numCol, source, where, title, yLabel, catSql = GROUP_ALIAS, catLabel = '',
  categoriesOrder = ctx.groups, catLabelFn = ctx.groupLabel,
}) {
  const single = await isOnePerCategory(ctx, numCol, source, where, catSql);
  return ctx.plot.engine.renderDistribution(slot, ctx, {
    numCol,
    source: { table: source, where },
    catSql,
    catLabel,
    yLabel,
    title,
    force: single ? 'bar' : 'auto',
    showSignificance: !single && !!ctx.state.showSignificance,
    series: { isCategory: true },
    categoriesOrder,
    catLabelFn,
  });
}

function plotGrid(container, perRow) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:12px';
  container.appendChild(wrap);
  return (slotStyle = '') => {
    const slot = document.createElement('div');
    slot.style.cssText =
      `flex:0 0 calc(${100 / perRow}% - 12px);min-width:300px;box-sizing:border-box;${slotStyle}`;
    wrap.appendChild(slot);
    return slot;
  };
}

/** A box summary per group for one metric: the overview tile's preview.
 *
 * A widget without this shows a placeholder icon in the tile grid. maxRawPoints:0 forces
 * the SQL box summary, because a preview has to stay cheap.
 */
function miniDistribution(ctx, container, { numCol, source, where, yLabel }) {
  return ctx.plot.engine.renderDistribution(container, ctx, {
    numCol,
    source: { table: source, where },
    catSql: GROUP_ALIAS,
    yLabel,
    series: { isCategory: true },
    categoriesOrder: ctx.groups,
    catLabelFn: ctx.groupLabel,
    mini: true,
    maxRawPoints: 0,
  });
}

// ── Instance morphology ───────────────────────────────────────────────────────

const instanceMorphology = {
  id: 'anatomy-instance-morphology',
  label: 'Instance Morphology',
  group: 'Dataset Stats',
  info: [
    'Every instance of the chosen structure, measured on its own: one violin per group, so a '
    + 'shift between violins is a difference between conditions rather than between objects.',
    '',
    'Turn on **Show significance** in the sidebar for Mann-Whitney brackets, Bonferroni '
    + 'corrected across the pairs shown. A metric with one value per group is drawn as bars and '
    + 'not tested, and a metric measured for only some instances says so.',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.includes('instance_entity');
  },

  async overviewPlot(container, ctx) {
    const metric = METRIC_ORDER.find((c) => ctx.schema.allCols.includes(c));
    const [entity] = await distinctValues(ctx, 'unnest(instance_entity)');
    if (!metric || !entity) return false;
    return miniDistribution(ctx, container, {
      numCol: metric,
      source: unnestedSource(ctx, ['instance_entity', metric]),
      where: `WHERE ${ctx.sql.q('instance_entity')} = ${esc(entity)}`,
      yLabel: labelFor(metric),
    });
  },

  async overviewMessage(ctx) {
    // Whichever size column this report has: unnesting an absent one is an error.
    const size = METRIC_ORDER.find((c) => ctx.schema.allCols.includes(c));
    if (!size) return null;
    const [row] = await ctx.queryRows(
      `SELECT COUNT(*) AS n, COUNT(DISTINCT ${ctx.sql.q('instance_entity')}) AS entities
       FROM ${unnestedSource(ctx, ['instance_entity', size])}`
    );
    if (!row || !Number(row.n)) return null;
    return `<strong>${Number(row.n).toLocaleString()}</strong> instances across `
      + `<strong>${row.entities}</strong> structure(s), each measured on its own.`;
  },

  async render(container, ctx) {
    const metrics = METRIC_ORDER.filter((c) => ctx.schema.allCols.includes(c));
    const entities = await distinctValues(ctx, 'unnest(instance_entity)');
    if (!entities.length || !metrics.length) {
      emptyState(container, 'No labelled instances in this report.');
      return;
    }

    const controls = document.createElement('div');
    controls.style.cssText = 'margin-bottom:.5rem';
    container.appendChild(controls);
    const plots = document.createElement('div');
    container.appendChild(plots);

    let entity = focusedStructure(entities);
    const draw = async () => {
      plots.replaceChildren();
      const source = unnestedSource(ctx, ['instance_entity', ...metrics]);
      const where = `WHERE ${ctx.sql.q('instance_entity')} = ${esc(entity)}`;
      const counts = await measuredMetrics(ctx, metrics, source, where);
      const drawable = metrics.filter((m) => counts[m] > 0);
      const [{ n }] = await ctx.queryRows(`SELECT COUNT(*) AS n FROM ${source} ${where}`);
      const nextSlot = plotGrid(plots, ctx.groups.length <= 2 ? 3 : 2);
      for (const metric of drawable) {
        await drawDistribution(ctx, nextSlot(), {
          numCol: metric,
          source,
          where,
          yLabel: labelFor(metric),
          title: labelFor(metric),
        });
      }
      // Partly measured is a warning: the plot shows a subset and does not say so. Not
      // measured at all is not. There is no plot, and the reason is worth one line.
      ctx.plot.dataAvailabilityWarning(
        plots,
        drawable.map((m) => ({ label: labelFor(m), present: counts[m] })),
        Number(n),
        { unit: `${entity} instances`, level: 'yellow' },
      );
      const absent = metrics.filter((m) => !counts[m]);
      if (absent.length) {
        caption(plots,
          `Not measured for <b>${entity}</b>: ${absent.map(labelFor).join(', ')}. Skeletons `
          + 'are skipped above <code>--max-skeleton-voxels</code>, and the distance to the '
          + 'nearest instance of the same structure needs a second one in the same object.');
      }
    };

    const picker = selector('Structure', entities, entity, (v) => {
      entity = v;
      focusStructure(v);
      draw();
    });
    controls.appendChild(picker);
    onStructureFocus(container, (name) => {
      if (name === entity || !entities.includes(name)) return;
      entity = name;
      picker.querySelector('select').value = name;
      draw();
    });
    await draw();
  },
};

// ── Distances between structures ──────────────────────────────────────────────

const instanceDistances = {
  id: 'anatomy-distances',
  label: 'Distances Between Structures',
  group: 'Dataset Stats',
  info: [
    'For each instance of the chosen structure, the smallest distance from its voxels to the '
    + 'named structure: how close it gets at its closest point.',
    '',
    'Measured voxel centre to voxel centre, so **one voxel step means touching** and **0 means '
    + 'the two overlap**. A violin pressed against the bottom is a population sitting on that '
    + 'structure; one lifted off it is a population keeping its distance.',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.includes('distance_um');
  },

  async overviewMessage(ctx) {
    const [row] = await ctx.queryRows(
      `SELECT COUNT(*) AS n,
              SUM(CASE WHEN ${ctx.sql.q('distance_um')} = 0 THEN 1 ELSE 0 END) AS overlapping
       FROM ${unnestedSource(ctx, ['distance_entity', 'distance_target', 'distance_um'])}`
    );
    if (!row || !Number(row.n)) return null;
    const share = (Number(row.overlapping) / Number(row.n)) * 100;
    return `<strong>${Number(row.n).toLocaleString()}</strong> instance-to-structure distances; `
      + `<strong>${share.toFixed(1)}%</strong> read 0, meaning the two overlap.`;
  },

  async overviewPlot(container, ctx) {
    return miniDistribution(ctx, container, {
      numCol: 'distance_um',
      source: unnestedSource(ctx, ['distance_entity', 'distance_target', 'distance_um']),
      where: '',
      yLabel: 'Distance (µm)',
    });
  },

  async render(container, ctx) {
    const pairs = await ctx.queryRows(
      `SELECT DISTINCT unnest(distance_entity) AS e, unnest(distance_target) AS t
       FROM pp_data ${ctx.sql.andWhere(ctx.where, OBJECT_ROW)} ORDER BY e, t`
    );
    if (!pairs.length) {
      emptyState(container, 'No distances in this report.');
      return;
    }

    const controls = document.createElement('div');
    controls.style.cssText = 'margin-bottom:.5rem';
    container.appendChild(controls);
    const plots = document.createElement('div');
    container.appendChild(plots);

    const entities = [...new Set(pairs.map((p) => p.e))];
    let entity = focusedStructure(entities);

    const draw = async () => {
      plots.replaceChildren();
      controls.replaceChildren();
      const targets = pairs.filter((p) => p.e === entity).map((p) => p.t);
      controls.appendChild(
        selector('Structure', entities, entity, (v) => {
          entity = v;
          focusStructure(v);
          draw();
        })
      );
      const source = unnestedSource(ctx, ['distance_entity', 'distance_target', 'distance_um']);
      const nextSlot = plotGrid(plots, targets.length <= 1 ? 1 : 2);
      for (const target of targets) {
        const where =
          `WHERE ${ctx.sql.q('distance_entity')} = ${esc(entity)}` +
          ` AND ${ctx.sql.q('distance_target')} = ${esc(target)}`;
        await drawDistribution(ctx, nextSlot(), {
          numCol: 'distance_um',
          source,
          where,
          yLabel: `Distance (µm)`,
          title: `${entity} → ${target}`,
        });
      }
    };

    await draw();
  },
};

// ── Reach: how close instances get to two structures at once ──────────────────

// Probabilities the reach curve is drawn from. A monotone curve needs no more
// vertices than this, so a group of 8000 instances costs 51 points, not 8000.
const CURVE_PROBABILITIES = Array.from({ length: 51 }, (_, i) => i / 50);

/** One row per (instance × pair of targets): the distance at which it reaches both. */
function reachSource(ctx, entity) {
  const long = unnestedSource(ctx, [
    'distance_entity', 'distance_label', 'distance_target', 'distance_um',
  ]);
  return `(
    SELECT a.${GROUP_ALIAS} AS grp, a."distance_target" AS target_a,
           b."distance_target" AS target_b,
           GREATEST(a."distance_um", b."distance_um") AS reach
    FROM ${long} a
    JOIN ${long} b
      ON a."object_id" = b."object_id"
     AND a."distance_entity" = b."distance_entity"
     AND a."distance_label" = b."distance_label"
     AND a."distance_target" < b."distance_target"
    WHERE a."distance_entity" = ${esc(entity)}
  )`;
}

const instanceReach = {
  id: 'anatomy-reach',
  label: 'Reaching Two Structures At Once',
  group: 'Dataset Stats',
  info: [
    'One panel per pair of structures, so every combination is readable at once. For each '
    + 'instance it takes the **larger** of its two distances: the distance at which it is close '
    + 'to **both**. The curve is the share of instances at or below it.',
    '',
    'A curve that **climbs early and steeply** means most instances sit against both; one '
    + '**pushed to the right** means few do. Every panel shares the same axes, and the curves '
    + 'are drawn from quantiles, so the reading is the same at any number of instances.',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.includes('distance_um');
  },

  async overviewMessage(ctx) {
    const [row] = await ctx.queryRows(
      `SELECT COUNT(DISTINCT ${ctx.sql.q('distance_target')}) AS targets
       FROM ${unnestedSource(ctx, ['distance_entity', 'distance_target', 'distance_um'])}`
    );
    const targets = Number(row?.targets ?? 0);
    if (targets < 2) return null;
    return `How close instances get to <strong>two</strong> structures at once, over `
      + `<strong>${(targets * (targets - 1)) / 2}</strong> pair(s) of structures.`;
  },

  async overviewPlot(container, ctx) {
    // One pair's curves, as a taste of the matrix the full widget draws.
    const [entity] = await distinctValues(ctx, 'unnest(distance_entity)');
    if (!entity) return false;
    const curves = await ctx.queryRows(`
      SELECT target_a, target_b, grp,
             quantile_cont(reach, [${CURVE_PROBABILITIES.join(', ')}]) AS quantiles
      FROM ${reachSource(ctx, entity)} WHERE reach IS NOT NULL
      GROUP BY 1, 2, 3 ORDER BY 1, 2, 3`);
    if (!curves.length) return false;
    const [first] = curves;
    ctx.plot.appendMini(container, curves
      .filter((r) => r.target_a === first.target_a && r.target_b === first.target_b)
      .map((r) => ({
        type: 'scatter', mode: 'lines',
        line: { shape: 'hv', width: 2, color: ctx.color.group(r.grp) },
        x: Array.from(r.quantiles ?? [], Number),
        y: CURVE_PROBABILITIES.map((prob) => prob * 100),
      })), { xaxis: { rangemode: 'tozero' }, yaxis: { ticksuffix: '%', range: [0, 102] } });
    return true;
  },

  async render(container, ctx) {
    const entities = await distinctValues(ctx, 'unnest(distance_entity)');
    if (!entities.length) {
      emptyState(container, 'No distances in this report.');
      return;
    }

    const controls = document.createElement('div');
    controls.style.cssText = 'margin-bottom:.5rem';
    container.appendChild(controls);
    const plots = document.createElement('div');
    container.appendChild(plots);

    let entity = focusedStructure(entities);
    const draw = async () => {
      plots.replaceChildren();
      const probs = `[${CURVE_PROBABILITIES.join(', ')}]`;
      const curves = await ctx.queryRows(`
        SELECT target_a, target_b, grp, COUNT(*) AS n,
               quantile_cont(reach, ${probs}) AS quantiles
        FROM ${reachSource(ctx, entity)}
        WHERE reach IS NOT NULL
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 3`);
      if (!curves.length) {
        emptyState(plots, `${entity} instances have fewer than two other structures to reach.`);
        return;
      }

      const asNumbers = (q) => Array.from(q ?? [], Number);
      // A shared X range makes every panel directly comparable.
      const xMax = Math.max(...curves.flatMap((r) => asNumbers(r.quantiles)));
      const pairs = [...new Set(curves.map((r) => `${r.target_a}\x00${r.target_b}`))];
      const nextSlot = plotGrid(plots, pairs.length <= 1 ? 1 : 2);

      for (const pair of pairs) {
        const [a, b] = pair.split('\x00');
        const traces = curves
          .filter((r) => r.target_a === a && r.target_b === b)
          .map((r) => ({
            type: 'scatter',
            mode: 'lines',
            line: { shape: 'hv', width: 2, color: ctx.color.group(r.grp) },
            name: `${ctx.groupLabel(r.grp)} (n=${Number(r.n).toLocaleString()})`,
            x: asNumbers(r.quantiles),
            y: CURVE_PROBABILITIES.map((p) => p * 100),
            hovertemplate: 'reaches both within %{x:.3g} µm<br>%{y:.0f}% of instances<extra>%{fullData.name}</extra>',
          }));
        ctx.plot.append(nextSlot(), traces, {
          title: { text: `${a} & ${b}` },
          xaxis: { title: 'reaches both within (µm)', rangemode: 'tozero', range: [0, xMax] },
          yaxis: { title: `% of ${entity} instances`, ticksuffix: '%', range: [0, 102] },
          showlegend: true,
          legend: { orientation: 'h', y: -0.25 },
        });
      }
    };

    const picker = selector('Structure', entities, entity, (v) => {
      entity = v;
      focusStructure(v);
      draw();
    });
    controls.appendChild(picker);
    onStructureFocus(container, (name) => {
      if (name === entity || !entities.includes(name)) return;
      entity = name;
      picker.querySelector('select').value = name;
      draw();
    });
    await draw();
  },
};


// ── Contacts & groups ─────────────────────────────────────────────────────────
//
// Three blocks, three different questions, and only the first one carries a test:
//
//   1. Is there more contact in one condition than in the other?  One point per object,
//      condition on the X axis. Objects are the replicates, so this is the only comparison a
//      p-value can be about.
//   2. What does a connected group reach?  A group counted as one thing, against what a group
//      of its size reaches by arithmetic alone. A relationship inside each condition, so
//      nothing here is tested either.
//   3. What touches what?  Composition, which is counted rather than compared.
//
// Everything reads the same selection: contacts of a chosen pair type, no wider than the gap
// threshold.

// Enough slider steps to feel continuous without re-querying on every pixel.
const GAP_STEPS = 40;
const MAX_EDGES = 400000;
const MAX_DISTANCE_ROWS = 500000;

const ALL_CONTACTS = 'all contacts';
const SAME_TYPE = 'same-type contacts';

/** Every recorded contact, one row per instance pair. */
function contactEdgeSource(ctx) {
  return unnestedSource(ctx, [
    'contact_entity_a', 'contact_label_a', 'contact_entity_b', 'contact_label_b',
    'contact_gap_um',
  ]);
}

const pairLabel = (a, b) => [a, b].sort().join(' + ');

/**
 * The contact selection: a SQL condition, and the structures it concerns.
 *
 * The structures matter because they are the denominator. Asked about mito + granules, "how
 * many instances are in contact" has to be out of the mito and granule instances, not out of
 * everything in the object.
 */
function selection(ctx, choice, gapUm) {
  const a = ctx.sql.q('contact_entity_a');
  const b = ctx.sql.q('contact_entity_b');
  const conditions = [`${ctx.sql.q('contact_gap_um')} <= ${gapUm}`];
  if (choice === SAME_TYPE) conditions.push(`${a} = ${b}`);
  if (choice === SAME_TYPE || choice === ALL_CONTACTS) {
    return { where: conditions.join(' AND '), structures: [] };
  }
  const [x, y] = choice.split(' + ');
  conditions.push(
    `((${a} = ${esc(x)} AND ${b} = ${esc(y)}) OR (${a} = ${esc(y)} AND ${b} = ${esc(x)}))`
  );
  return { where: conditions.join(' AND '), structures: [...new Set([x, y])] };
}

/** The contacts the selection keeps. */
function selectedEdges(ctx, where) {
  return `(SELECT * FROM ${contactEdgeSource(ctx)} WHERE ${where})`;
}

/**
 * How many distinct partners each instance has in the selection.
 *
 * Both sides of every contact, since an instance can land in either column, and DISTINCT in
 * case a pair was recorded twice.
 */
function partnerCountSql(ctx, edges) {
  const q = ctx.sql.q;
  const side = (entity, label, otherEntity, otherLabel) =>
    `SELECT "object_id", ${q(entity)} AS "entity", ${q(label)} AS "label",
            ${q(otherEntity)} || ':' || CAST(${q(otherLabel)} AS VARCHAR) AS "partner"
     FROM ${edges}`;
  return `(SELECT "object_id", "entity", "label", COUNT(DISTINCT "partner") AS "partners"
           FROM (${side('contact_entity_a', 'contact_label_a',
                        'contact_entity_b', 'contact_label_b')}
                 UNION ALL
                 ${side('contact_entity_b', 'contact_label_b',
                        'contact_entity_a', 'contact_label_a')})
           GROUP BY 1, 2, 3)`;
}

/**
 * One row per instance the selection concerns, with its partner count.
 *
 * Instances without a single contact stay in, at zero partners. They are what makes "how many
 * are in contact" a real fraction rather than always 100%.
 */
function instanceProfileSql(ctx, { edges, structures }) {
  const q = ctx.sql.q;
  const instances = unnestedSource(ctx, ['instance_entity', 'instance_label']);
  const inPlay = structures.length
    ? `WHERE i.${q('instance_entity')} IN (${structures.map(esc).join(', ')})`
    : '';
  return `(SELECT i."object_id" AS "object_id", i.${GROUP_ALIAS} AS ${GROUP_ALIAS},
             COALESCE(p."partners", 0) AS "partners"
      FROM ${instances} i
      LEFT JOIN ${partnerCountSql(ctx, edges)} p
             ON p."object_id" = i."object_id"
            AND p."entity" = i.${q('instance_entity')}
            AND p."label" = i.${q('instance_label')}
      ${inPlay})`;
}

/** Instances and contact share, per object. */
function perObjectSql(ctx, profile) {
  return `SELECT "object_id", ANY_VALUE(${GROUP_ALIAS}) AS "grp", COUNT(*) AS "instances",
            COUNT(*) FILTER (WHERE "partners" > 0) AS "in_contact",
            AVG("partners") AS "mean_partners"
     FROM ${profile} GROUP BY 1`;
}

/** Contacts and their typical gap, per object. */
function perObjectGapSql(ctx, edges) {
  return `SELECT "object_id", COUNT(*) AS "contacts",
            MEDIAN(${ctx.sql.q('contact_gap_um')}) AS "median_gap"
     FROM ${edges} GROUP BY 1`;
}

/**
 * A source table built from rows this widget worked out itself.
 *
 * Cluster sizes come out of a union-find over the edge list, not out of SQL, and a chart drawn
 * by hand gets none of the engine's violins, palette or significance. Handing the rows back as
 * a table expression puts them through the same engine as every other object-level metric in
 * the report. One row per object, so it stays small.
 */
function valuesTable(rows, columns) {
  const literal = (value) => {
    if (value === null || value === undefined) return 'NULL';
    if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'NULL';
    return esc(String(value));
  };
  const tuples = rows.map((row) => `(${columns.map((c) => literal(row[c.key])).join(', ')})`);
  const names = columns.map((c) => `"${c.key}"`).join(', ');
  const casts = columns.map((c) => `CAST("${c.key}" AS ${c.type}) AS "${c.key}"`).join(', ');
  return `(SELECT ${casts} FROM (VALUES ${tuples.join(', ')}) AS t(${names}))`;
}

// What one object's contacts amount to. Rates and typical values, so objects with different
// instance counts stay comparable; the counts behind them are in the table at the bottom.
const OBJECT_METRICS = [
  { key: 'pct_in_contact', title: 'Instances in contact', axis: '% of instances' },
  { key: 'mean_partners', title: 'Partners per instance', axis: 'partners' },
  { key: 'largest_cluster', title: 'Largest cluster', axis: 'instances' },
  { key: 'median_gap', title: 'Typical gap', axis: 'median gap (µm)' },
];

const OBJECT_COLUMNS = [
  { key: GROUP_KEY, type: 'VARCHAR' },
  ...OBJECT_METRICS.map((m) => ({ key: m.key, type: 'DOUBLE' })),
];

/**
 * The condition comparison: one point per object, one panel per metric.
 *
 * Objects are the replicates. Instances inside an object are not independent of each other, so
 * a test over instances would find a difference in any two conditions and mean nothing; a test
 * over objects is the one that answers "is there more contact here than there". Same axis
 * layout across the panels, so the four read as one figure.
 */
async function objectComparison(ctx, container, { rows }) {
  const nextSlot = plotGrid(container, OBJECT_METRICS.length);
  const missing = [];
  for (const metric of OBJECT_METRICS) {
    const usable = rows.filter((row) => row[metric.key] !== null);
    if (!usable.length) {
      missing.push(metric.title.toLowerCase());
      continue;
    }
    await drawDistribution(ctx, nextSlot('min-width:230px'), {
      numCol: metric.key,
      source: valuesTable(usable, OBJECT_COLUMNS),
      where: '',
      title: metric.title,
      yLabel: metric.axis,
      categoriesOrder: ctx.groups.map(String),
    });
  }
  return missing;
}

const median = (values) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/**
 * Every instance's distance to every structure.
 *
 * Distances do not depend on the contact selection, so this is read once and the whole block is
 * JavaScript from there.
 */
function clusterReachSql(ctx) {
  const q = ctx.sql.q;
  return `SELECT "object_id", ${q('distance_entity')} AS "entity",
            ${q('distance_label')} AS "label", ${q('distance_target')} AS "target",
            ${q('distance_um')} AS "distance"
     FROM ${unnestedSource(ctx, ['distance_entity', 'distance_label', 'distance_target',
                                 'distance_um'])}`;
}

/** How many of `sorted` are below `value`, or at most `value`. */
function countUpTo(sorted, value, inclusive) {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (inclusive ? sorted[mid] <= value : sorted[mid] < value) low = mid + 1;
    else high = mid;
  }
  return low;
}

/**
 * Where a group's reach falls among the random groups of the same size and make-up.
 *
 * A group of ten reaches closer than a group of two whatever it is made of, because the closest
 * of ten draws is closer than the closest of two. So the size correction goes inside the number
 * rather than onto an axis: what comes out is the share of comparable groups, drawn from the
 * instances the group is made of in the object it lives in, that this group beats. Half is
 * chance.
 *
 * P(a random group reaches no closer) = C(n - r, k) / C(n, k), with r the instances at or nearer
 * than `observed`. Averaged with the same term for the ones strictly nearer, so ties split
 * evenly and the result is flat between 0 and 1 when nothing is going on.
 */
function nullPercentile(sorted, k, observed) {
  const n = sorted.length;
  if (!n || k >= n) return null;      // the group is everything there is: nothing to compare
  const survival = (r) => {
    let p = 1;
    for (let j = 0; j < k; j++) {
      if (n - r - j <= 0) return 0;
      p *= (n - r - j) / (n - j);
    }
    return p;
  };
  const atOrNearer = countUpTo(sorted, observed, true);
  const nearer = countUpTo(sorted, observed, false);
  return ((survival(atOrNearer) + survival(nearer)) / 2) * 100;
}

/**
 * One record per group and structure it could reach: how close it gets, and how that compares
 * with the groups chance would give.
 *
 * A group counts as one thing, so it reaches a structure as far as its closest member does. What
 * a group is made of comes from the contact selection: same-type contacts chain one structure, a
 * pair chains two. A structure the group already holds is left out, since that would measure the
 * group against itself, and groups of one are left out because they are instances rather than
 * groups.
 */
function clusterReach({ rows, groups, grpOf }) {
  const spread = new Map();     // object, structure, target -> every distance there is
  const closest = new Map();    // group -> target -> the closest one member gets
  for (const row of rows) {
    const object = String(row.object_id);
    const distance = Number(row.distance);
    if (!Number.isFinite(distance)) continue;

    const key = `${object}\x00${row.entity}\x00${row.target}`;
    const all = spread.get(key) ?? [];
    all.push(distance);
    spread.set(key, all);

    const root = groups.rootOf(instanceKey(object, row.entity, row.label));
    if (root === undefined || (groups.sizes.get(root) ?? 0) < 2) continue;
    const perGroup = closest.get(root) ?? new Map();
    const seen = perGroup.get(row.target);
    if (seen === undefined || distance < seen) perGroup.set(row.target, distance);
    closest.set(root, perGroup);
  }
  for (const all of spread.values()) all.sort((a, b) => a - b);

  // The instances a group of this make-up could have been drawn from. Cached, because there are
  // a handful of make-ups and one merge each is enough.
  const pools = new Map();
  const poolFor = (object, made, target) => {
    const key = `${object}\x00${made.join(',')}\x00${target}`;
    let pool = pools.get(key);
    if (!pool) {
      pool = made
        .flatMap((structure) => spread.get(`${object}\x00${structure}\x00${target}`) ?? [])
        .sort((a, b) => a - b);
      pools.set(key, pool);
    }
    return pool;
  };

  const records = [];
  for (const [root, targets] of closest) {
    const size = groups.sizes.get(root) ?? 0;
    const made = [...(groups.rootEntities.get(root) ?? [])].sort();
    const object = String(groups.rootObject.get(root));
    for (const [target, observed] of targets) {
      if (made.includes(target)) continue;
      const beats = nullPercentile(poolFor(object, made, target), size, observed);
      if (beats === null) continue;
      records.push({
        grp: String(grpOf.get(object) ?? ''), group: String(root), target, size,
        made: made.join(' + '), observed, beats,
      });
    }
  }
  return records;
}

/**
 * What connected groups reach, against the groups chance would give.
 *
 * One box per condition and structure. Half is where a group lands when its reach is whatever its
 * size implies, so the reading is the distance from that line and nothing else. The concrete
 * distance is under each label, because "beats 70% of them" still leaves open whether that means
 * touching or a micron away.
 */
function reachAgainstChance(ctx, container, { records }) {
  if (!records.length) return false;

  const targets = [...new Set(records.map((r) => r.target))];
  const medianFor = (target, pick) => median(
    records.filter((r) => r.target === target).map(pick));
  const reachOf = new Map(targets.map((t) => [t, medianFor(t, (r) => r.observed)]));
  // Strongest first: the structures the groups are actually placed against.
  const order = [...targets].sort(
    (a, b) => medianFor(b, (r) => r.beats) - medianFor(a, (r) => r.beats));
  const present = ctx.groups.filter((g) => records.some((r) => r.grp === String(g)));
  const shown = present.length ? present : [...new Set(records.map((r) => r.grp))];

  const traces = shown.map((condition) => {
    const mine = records.filter((r) => r.grp === String(condition));
    const counted = new Set(mine.map((r) => r.group)).size;
    return {
      type: 'box',
      name: `${ctx.groupLabel(condition)} (${counted.toLocaleString()} groups)`,
      x: mine.map((r) => r.target),
      y: mine.map((r) => r.beats),
      customdata: mine.map((r) => [r.size, r.made]),
      boxpoints: mine.length <= 200 ? 'all' : 'outliers',
      jitter: 0.4,
      pointpos: 0,
      marker: { color: ctx.color.group(condition), size: 4, opacity: 0.5 },
      line: { color: ctx.color.group(condition) },
      hovertemplate: '%{x}: beats %{y:.0f}% of comparable groups'
        + '<br>group of %{customdata[0]} (%{customdata[1]})<extra></extra>',
    };
  }).filter((trace) => trace.y.length);
  if (!traces.length) return false;

  ctx.plot.append(container, traces, {
    boxmode: 'group',
    xaxis: {
      type: 'category',
      categoryarray: order,
      tickvals: order,
      ticktext: order.map((target) => {
        const reach = reachOf.get(target);
        return `${target}<br><span style="font-size:.8em;color:#777">`
          + `${reach ? `${reach.toFixed(3)} µm` : 'touching'}</span>`;
      }),
      automargin: true,
    },
    yaxis: {
      title: 'closer than % of comparable random groups',
      range: [-2, 102],
      ticksuffix: '%',
    },
    shapes: [{
      type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 50, y1: 50,
      line: { color: '#888', width: 1, dash: 'dash' },
    }],
    annotations: [{
      xref: 'paper', x: 1, y: 50, yref: 'y', text: 'chance', showarrow: false,
      xanchor: 'left', font: { size: 10, color: '#888' },
    }],
    showlegend: shown.length > 1,
  });
  return true;
}

/**
 * Per condition and structure: how much it touches, and how that splits by partner.
 *
 * A mixed pair is a contact for both structures, a self pair is one contact for one structure.
 * Every structure's partners therefore add up to its total, which is what makes the shares
 * below add up to 100%.
 */
function partnerShares(rows) {
  const byGroup = new Map();
  for (const row of rows) {
    const grp = String(row.grp);
    const n = Number(row.n);
    const sides = row.s1 === row.s2
      ? [[row.s1, row.s2]]
      : [[row.s1, row.s2], [row.s2, row.s1]];
    const structures = byGroup.get(grp) ?? new Map();
    for (const [structure, partner] of sides) {
      const own = structures.get(structure) ?? { total: 0, partners: new Map() };
      own.total += n;
      own.partners.set(partner, (own.partners.get(partner) ?? 0) + n);
      structures.set(structure, own);
    }
    byGroup.set(grp, structures);
  }
  return byGroup;
}

/**
 * What each structure's contacts are with, as a share of its own contacts.
 *
 * Row-normalised, so a row reads "of everything this structure touches, this much is X", and
 * the diagonal is its own kind, which is what makes long chains of one structure. Without the
 * normalisation the panel would only show which structure is the most numerous. Rows are
 * ordered by how much contact the structure is in, so the busiest one is at the top.
 */
function partnerMix(ctx, container, { rows, structures }) {
  if (!rows.length) return false;

  const shares = partnerShares(rows);
  const contactsOf = (name) => [...shares.values()].reduce(
    (sum, per) => sum + (per.get(name)?.total ?? 0), 0);
  const busiest = [...structures].sort((a, b) => contactsOf(b) - contactsOf(a));
  const present = ctx.groups.filter((g) => shares.has(String(g)));
  const shown = present.length ? present : [...shares.keys()];

  const nextSlot = plotGrid(container, Math.min(shown.length, 2));
  for (const group of shown) {
    const of = shares.get(String(group)) ?? new Map();
    const pairsOf = (row, column) => of.get(row)?.partners.get(column) ?? 0;
    const share = busiest.map((row) => {
      const total = of.get(row)?.total ?? 0;
      return busiest.map((column) => (total ? (pairsOf(row, column) / total) * 100 : null));
    });
    const annotations = [];
    share.forEach((values, r) => values.forEach((value, c) => {
      if (value === null || value < 1) return;      // an empty cell reads better than "0%"
      annotations.push({
        x: busiest[c], y: busiest[r], text: `${Math.round(value)}%`, showarrow: false,
        font: { size: 10, color: value > 55 ? '#fff' : '#333' },
      });
    }));

    ctx.plot.append(nextSlot(), [{
      type: 'heatmap',
      x: busiest,
      y: busiest,
      z: share,
      customdata: busiest.map((row) => busiest.map((column) => pairsOf(row, column))),
      zmin: 0,
      zmax: 100,
      colorscale: 'YlGnBu',
      reversescale: true,
      hoverongaps: false,
      hovertemplate:
        '%{z:.1f}% of what %{y} touches is %{x}<br>%{customdata:,} pair(s)<extra></extra>',
      colorbar: { title: '% of row', ticksuffix: '%' },
    }], {
      title: { text: `What ${ctx.groupLabel(group)} touches` },
      xaxis: { type: 'category', side: 'top', automargin: true },
      yaxis: { type: 'category', automargin: true, autorange: 'reversed' },
      annotations,
      height: Math.max(240, 46 * busiest.length + 140),
    });
  }
  return true;
}

/** Which structures touch which, counted once per unordered pair. */
function pairCountsSql(ctx, edges) {
  const a = ctx.sql.q('contact_entity_a');
  const b = ctx.sql.q('contact_entity_b');
  return `SELECT ${GROUP_ALIAS} AS grp,
            LEAST(${a}, ${b}) AS s1, GREATEST(${a}, ${b}) AS s2, COUNT(*) AS n
     FROM ${edges} GROUP BY 1, 2, 3`;
}

/** One instance's key in a contact group: object, structure and label, all stringified. */
function instanceKey(object, entity, label) {
  // String() every part: DuckDB hands int64 columns over as BigInt, which JSON.stringify
  // refuses outright ("Do not know how to serialize a BigInt").
  return JSON.stringify([String(object), String(entity), String(label)]);
}

export { contactEdgeSource, pairCountsSql, unnestedSource, reachSource };
export { instanceProfileSql, perObjectSql, perObjectGapSql, clusterReachSql, valuesTable };

// For the widget harness: the charts, so a test can draw them and catch what only shows up at
// run time. They take a container and a ctx and touch nothing else.
export { partnerMix, partnerShares, objectComparison };
export { clusterReach, nullPercentile, reachAgainstChance };

/**
 * Connected components of instances linked by contacts: a "cluster".
 *
 * Instances are seeded as singletons, so the denominator is every instance that could have
 * touched. Identity is object + entity + label, since a label id repeats across objects and
 * structures; the object and the structure are kept in their own maps rather than parsed back
 * out of a joined key, because folder names may contain anything.
 */
export function contactGroups(instances, edges) {
  const parent = new Map();
  const objectOf = new Map();
  const entityOf = new Map();
  const find = (x) => {
    while (parent.get(x) !== x) {
      parent.set(x, parent.get(parent.get(x)));
      x = parent.get(x);
    }
    return x;
  };
  const ensure = (object, entity, label) => {
    const k = instanceKey(object, entity, label);
    if (!parent.has(k)) {
      parent.set(k, k);
      objectOf.set(k, object);
      entityOf.set(k, entity);
    }
    return k;
  };

  for (const i of instances) ensure(i.object_id, i.entity, i.label);
  for (const e of edges) {
    const a = ensure(e.object_id, e.entity_a, e.label_a);
    const b = ensure(e.object_id, e.entity_b, e.label_b);
    parent.set(find(a), find(b));
  }

  const sizes = new Map();
  const rootObject = new Map();
  const rootEntities = new Map();
  for (const k of parent.keys()) {
    const root = find(k);
    sizes.set(root, (sizes.get(root) ?? 0) + 1);
    if (!rootObject.has(root)) rootObject.set(root, objectOf.get(k));
    const seen = rootEntities.get(root) ?? new Set();
    seen.add(entityOf.get(k));
    rootEntities.set(root, seen);
  }
  return {
    sizes,
    rootObject,
    rootEntities,
    total: parent.size,
    /** Which group an instance landed in, for anything that joins back to it. */
    rootOf: (key) => (parent.has(key) ? find(key) : undefined),
  };
}

/**
 * Clusters per object: the ones of two or more, since a lone instance is not a cluster.
 *
 * Per object rather than per condition, because that is the unit the charts compare. `mixed`
 * counts the clusters holding more than one structure, which is the difference between a chain
 * of one structure and a pile of several.
 */
export function clustersByObject(groups) {
  const byObject = new Map();
  for (const [root, size] of groups.sizes) {
    if (size < 2) continue;
    const object = String(groups.rootObject.get(root));
    const entry = byObject.get(object)
      ?? { clusters: 0, instances: 0, largest: 0, mixed: 0 };
    entry.clusters += 1;
    entry.instances += size;
    entry.largest = Math.max(entry.largest, size);
    if ((groups.rootEntities.get(root)?.size ?? 1) > 1) entry.mixed += 1;
    byObject.set(object, entry);
  }
  return byObject;
}

const contactsAndGroups = {
  id: 'anatomy-contacts',
  label: 'Contacts & Groups',
  group: 'Dataset Stats',
  info: [
    'A **contact** is a pair of instances whose surfaces come within the gap threshold of each '
    + 'other. A **cluster** is a set of instances chained together by them, so a cluster of five '
    + 'is five instances in contact through some path. Drag the threshold down to break the '
    + 'clusters apart and see which contacts survive.',
    '',
    'Three blocks, three questions, and only the first carries a test:',
    '',
    '- **Is there more contact in one condition?** One point per object. Objects are what a '
    + 'condition has several of, so this is the only comparison a p-value can be about.',
    '- **What does a connected group reach?** The group counted as one thing, against the '
    + 'groups of its own size that chance would give. A relationship, not a group comparison.',
    '- **What is each structure in contact with?** Composition, counted rather than compared.',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.includes('contact_gap_um');
  },

  async overviewPlot(container, ctx) {
    return miniDistribution(ctx, container, {
      numCol: 'contact_gap_um',
      source: contactEdgeSource(ctx),
      where: '',
      yLabel: 'Gap (µm)',
    });
  },

  async overviewMessage(ctx) {
    const [row] = await ctx.queryRows(
      `SELECT COUNT(*) AS n, MAX(${ctx.sql.q('contact_gap_um')}) AS widest
       FROM ${contactEdgeSource(ctx)}`
    );
    if (!row || !Number(row.n)) return null;
    return `<strong>${Number(row.n).toLocaleString()}</strong> instance pairs lie within `
      + `<strong>${Number(row.widest).toFixed(3)} µm</strong> of each other.`;
  },

  async render(container, ctx) {
    const edgeSource = contactEdgeSource(ctx);
    const [limits] = await ctx.queryRows(
      `SELECT COUNT(*) AS n, MAX(${ctx.sql.q('contact_gap_um')}) AS widest FROM ${edgeSource}`
    );
    if (!limits || !Number(limits.n)) {
      emptyState(container, 'No contacts in this report.');
      return;
    }
    const widest = Number(limits.widest) || 0;

    const pairs = await ctx.queryRows(
      `SELECT DISTINCT ${ctx.sql.q('contact_entity_a')} AS a, ${ctx.sql.q('contact_entity_b')} AS b
       FROM ${edgeSource} ORDER BY 1, 2`
    );
    const structures = [...new Set(pairs.flatMap((p) => [p.a, p.b]))].sort();
    const pairOptions = [
      ALL_CONTACTS, SAME_TYPE, ...new Set(pairs.map((p) => pairLabel(p.a, p.b))),
    ];

    // Every instance, so an instance that touches nothing still counts as a cluster of one.
    const instances = (await ctx.queryRows(
      `SELECT ${ctx.sql.q('object_id')} AS object_id, ${ctx.sql.q('instance_entity')} AS entity,
              ${ctx.sql.q('instance_label')} AS label
       FROM ${unnestedSource(ctx, ['instance_entity', 'instance_label'])}`
    )).map((r) => ({ object_id: String(r.object_id), entity: r.entity, label: r.label }));

    // Which condition each object belongs to. Groups are worked out per object, and this is
    // what puts a group in a condition.
    const grpOf = new Map((await ctx.queryRows(
      `SELECT ${ctx.sql.q('object_id')} AS object_id, ${ctx.sql.groupCol()} AS grp
       FROM pp_data ${ctx.sql.andWhere(ctx.where, OBJECT_ROW)}`
    )).map((r) => [String(r.object_id), String(r.grp)]));

    const controls = document.createElement('div');
    controls.style.cssText =
      'display:flex;align-items:center;flex-wrap:wrap;gap:1rem;margin-bottom:.5rem';
    container.appendChild(controls);
    const intro = document.createElement('div');
    const comparison = document.createElement('div');
    const reach = document.createElement('div');
    const mix = document.createElement('div');
    const numbers = document.createElement('div');
    for (const panel of [intro, comparison, reach, mix, numbers]) container.appendChild(panel);

    // Read once: distances do not move when the contact selection does.
    const distanceRows = await ctx.queryRows(
      `${clusterReachSql(ctx)} LIMIT ${MAX_DISTANCE_ROWS}`);

    let choice = pairOptions[0];
    let gapUm = widest;

    const draw = async () => {
      for (const panel of [intro, comparison, mix, numbers]) panel.replaceChildren();

      const { where, structures: inPlay } = selection(ctx, choice, gapUm);
      const edges = selectedEdges(ctx, where);
      const profile = instanceProfileSql(ctx, { edges, structures: inPlay });

      const objectRows = await ctx.queryRows(perObjectSql(ctx, profile));
      const gapRows = await ctx.queryRows(perObjectGapSql(ctx, edges));
      const edgeRows = await ctx.queryRows(
        `SELECT ${ctx.sql.q('object_id')} AS object_id,
                ${ctx.sql.q('contact_entity_a')} AS entity_a,
                ${ctx.sql.q('contact_label_a')} AS label_a,
                ${ctx.sql.q('contact_entity_b')} AS entity_b,
                ${ctx.sql.q('contact_label_b')} AS label_b
         FROM ${edges} LIMIT ${MAX_EDGES}`
      );
      if (edgeRows.length >= MAX_EDGES) {
        ctx.plot.prependWarning(intro, {
          html: `Only the first ${MAX_EDGES.toLocaleString()} contacts were clustered, so the `
            + 'cluster numbers are a lower bound. Lower the gap threshold for the real ones.',
        });
      }

      const groups = contactGroups(
        instances, edgeRows.map((e) => ({ ...e, object_id: String(e.object_id) })));
      const clusters = clustersByObject(groups);
      const gapOf = new Map(gapRows.map((r) => [String(r.object_id), r]));

      const perObject = objectRows.map((row) => {
        const object = String(row.object_id);
        const cluster = clusters.get(object)
          ?? { clusters: 0, instances: 0, largest: 0, mixed: 0 };
        const gap = gapOf.get(object);
        const count = Number(row.instances);
        const inContact = Number(row.in_contact);
        return {
          [GROUP_KEY]: String(row.grp),
          object,
          instances: count,
          in_contact: inContact,
          contacts: gap ? Number(gap.contacts) : 0,
          clusters: cluster.clusters,
          mixed: cluster.mixed,
          pct_in_contact: count ? (inContact / count) * 100 : null,
          mean_partners: row.mean_partners === null ? null : Number(row.mean_partners),
          largest_cluster: cluster.largest || null,
          median_gap: gap ? Number(gap.median_gap) : null,
        };
      });

      const conditions = ctx.groups.filter(
        (g) => perObject.some((row) => row[GROUP_KEY] === String(g)));
      const shown = conditions.length
        ? conditions : [...new Set(perObject.map((r) => r[GROUP_KEY]))];

      heading(comparison, 'Is there more contact in one condition than in the other?');
      const missing = await objectComparison(ctx, comparison, { rows: perObject });
      caption(comparison,
        'One point per object. Objects are what a condition has several of, so this is the '
        + 'block a difference between conditions can be read off, and the only one with a '
        + 'significance test when significance is on. '
        + (missing.length
          ? `Not shown: ${missing.join(', ')}, nothing measured at this selection. ` : '')
        + 'The counts behind these rates are in the table at the bottom.');

      heading(reach, 'What does a connected group reach?');
      const records = distanceRows.length < MAX_DISTANCE_ROWS
        ? clusterReach({ rows: distanceRows, groups, grpOf }) : [];
      if (reachAgainstChance(ctx, reach, { records })) {
        caption(reach,
          'A group is whatever the contacts above chain together, counted as one thing: it '
          + 'reaches a structure as far as its closest member does, and that distance is under '
          + 'each label. Bigger groups reach closer whatever they are made of, since the closest '
          + 'of ten beats the closest of two, so the axis asks the fairer question. Out of the '
          + 'random groups of the same size, drawn from the same structures in the same object, '
          + 'how many does this group beat? Half is chance. Above the line the connected ones do '
          + 'sit closer than their size explains, below they sit further, and on the line being '
          + 'in a group says nothing about where an instance is.');
      } else if (choice === ALL_CONTACTS) {
        note(reach,
          'Nothing left for a group to reach: with every kind of contact counted, the groups here '
          + `hold every structure. Pick "${SAME_TYPE}" or one pair above, or lower the gap.`);
      } else {
        note(reach, 'No group of two or more at this selection.');
      }

      if (choice === ALL_CONTACTS && structures.length > 1) {
        heading(mix, 'What is each structure in contact with?');
        const pairRows = await ctx.queryRows(pairCountsSql(ctx, edges));
        if (partnerMix(ctx, mix, { rows: pairRows, structures })) {
          caption(mix,
            'Each row is one structure and adds up to 100%: of all the contacts it is in, how '
            + 'many are with which structure. The diagonal is its own kind, so a structure high '
            + 'on the diagonal chains with itself and one low on it only ever groups up against '
            + 'something else. Shares rather than counts, otherwise the panel would only show '
            + 'which structure is the most numerous. Hover for the pair count.');
        }
      } else if (structures.length > 1) {
        note(mix, `What touches what needs "${ALL_CONTACTS}": one pair on its own is 100% of `
          + 'itself.');
      }

      const byCondition = new Map();
      for (const row of perObject) {
        const entry = byCondition.get(row[GROUP_KEY]) ?? {
          objects: 0, instances: 0, in_contact: 0, contacts: 0, clusters: 0, mixed: 0, largest: 0,
        };
        entry.objects += 1;
        entry.instances += row.instances;
        entry.in_contact += row.in_contact;
        entry.contacts += row.contacts;
        entry.clusters += row.clusters;
        entry.mixed += row.mixed;
        entry.largest = Math.max(entry.largest, row.largest_cluster ?? 0);
        byCondition.set(row[GROUP_KEY], entry);
      }
      heading(numbers, 'The counts');
      numbers.appendChild(ctx.plot.statTable(
        ['', 'objects', 'instances', 'in contact', 'contacts', 'clusters', 'largest',
         'mixed clusters'],
        shown.map((condition) => {
          const e = byCondition.get(String(condition));
          return [
            ctx.groupLabel(condition),
            e.objects,
            e.instances.toLocaleString(),
            e.instances ? `${((e.in_contact / e.instances) * 100).toFixed(1)}%` : '-',
            e.contacts.toLocaleString(),
            e.clusters.toLocaleString(),
            e.largest || '-',
            e.clusters ? `${((e.mixed / e.clusters) * 100).toFixed(0)}%` : '-',
          ];
        })
      ));
      caption(numbers,
        'A mixed cluster holds more than one structure: the rest are chains of a single '
        + 'structure.');
    };

    controls.appendChild(selector('Contacts', pairOptions, choice, (value) => {
      choice = value;
      draw();
    }));
    const sliderWrap = document.createElement('label');
    sliderWrap.style.cssText =
      'display:inline-flex;align-items:center;gap:.4rem;font-size:.8rem';
    const readout = document.createElement('span');
    readout.textContent = `gap <= ${widest.toFixed(3)} µm`;
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = String(GAP_STEPS);
    slider.value = String(GAP_STEPS);
    slider.addEventListener('input', () => {
      gapUm = (Number(slider.value) / GAP_STEPS) * widest;
      readout.textContent = `gap <= ${gapUm.toFixed(3)} µm`;
    });
    // Redraw on release rather than on every pixel: each step is a query plus a union-find.
    slider.addEventListener('change', draw);
    sliderWrap.append('Gap threshold', slider, readout);
    controls.appendChild(sliderWrap);

    await draw();
  },
};

export default [instanceMorphology, instanceDistances, instanceReach, contactsAndGroups];
