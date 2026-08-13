/**
 * CellSketch viewer widgets.
 *
 * Per-instance data is stored as parallel list columns on the cell row (see the
 * package README), so both widgets build their own source: a subquery that unnests
 * those lists into one row per instance, handed to the viewer's own distribution
 * engine. That is what puts instance-level data through the same violin/box, palette,
 * grouping and Mann-Whitney significance machinery as the built-in widgets - the
 * engine takes an arbitrary source table expression, so it does not care that the
 * rows came from an unnest.
 *
 * Exported as an array: one file, two widgets, no cross-file imports to serve.
 */

const METRIC_LABELS = {
  instance_volume_um3: 'Volume (µm³)',
  instance_surface_area_um2: 'Surface area (µm²)',
  instance_sphericity: 'Sphericity',
  instance_aspect_ratio_major_minor: 'Aspect ratio (major/minor)',
  instance_branches: 'Skeleton branches',
  instance_length_um: 'Skeleton length (µm)',
  instance_tortuosity: 'Tortuosity',
  instance_distance_to_closest_same_type_um: 'Distance to nearest same type (µm)',
  instance_polar_dist_um: 'Distance from cell centre (µm)',
  instance_polar_az_deg: 'Azimuth from cell centre (°)',
  instance_polar_el_deg: 'Elevation from cell centre (°)',
};

// Curated order: size and shape first, then skeleton, then position.
const METRIC_ORDER = Object.keys(METRIC_LABELS);

const CELL_ROW = '"obs_level" = 0';
// Columns carrying text rather than a measurement, so the finite guard below skips them.
const TEXT_COLUMNS = new Set([
  'instance_entity', 'distance_entity', 'distance_target', 'distance_hist_counts',
  'contact_entity_a', 'contact_entity_b',
]);
const GROUP_ALIAS = '"__cs_group__"';

const esc = (v) => `'${String(v).replace(/'/g, "''")}'`;
const labelFor = (col) => METRIC_LABELS[col] ?? col;

/** Distinct values of a scalar SQL expression over the cell rows, in sorted order. */
async function distinctValues(ctx, expr) {
  const where = ctx.sql.andWhere(ctx.where, CELL_ROW);
  const rows = await ctx.queryRows(
    `SELECT DISTINCT ${expr} AS v FROM pp_data ${where} ORDER BY v`
  );
  return rows.map((r) => r.v).filter((v) => v !== null && v !== undefined);
}

/**
 * A source table expression yielding one row per unnested element.
 * Every listed column is unnested in the same SELECT, which keeps them row-aligned.
 */
function unnestedSource(ctx, columns) {
  const selects = columns.map((c) => `unnest(${ctx.sql.q(c)}) AS ${ctx.sql.q(c)}`);
  const where = ctx.sql.andWhere(ctx.where, CELL_ROW);
  // cell_id rides along: an instance is identified by cell + entity + label, which is
  // what any join on instances has to match.
  const inner = `(SELECT "cell_id", ${ctx.sql.groupCol()} AS ${GROUP_ALIAS}, ${selects.join(', ')}
                  FROM pp_data ${where})`;
  // A NaN in a measurement column makes DuckDB's STDDEV raise "out of range", and turns
  // min/max/quantiles into nan. The processors write NULL for anything they could not
  // measure, but reports written before that fix still hold NaNs - so guard here too.
  const guarded = columns.map((c) => {
    const q = ctx.sql.q(c);
    return TEXT_COLUMNS.has(c) ? q : `CASE WHEN isfinite(${q}) THEN ${q} END AS ${q}`;
  });
  return `(SELECT "cell_id", ${GROUP_ALIAS}, ${guarded.join(', ')} FROM ${inner})`;
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

function emptyState(container, text) {
  const p = document.createElement('p');
  p.style.cssText = 'font-size:.85rem;color:#666';
  p.textContent = text;
  container.appendChild(p);
}

/** One violin per group for `numCol`, over `source`, with significance if enabled. */
function drawDistribution(ctx, cell, { numCol, source, where, title, yLabel }) {
  return ctx.plot.engine.renderDistribution(cell, ctx, {
    numCol,
    source: { table: source, where },
    catSql: GROUP_ALIAS,
    catLabel: '',
    yLabel,
    title,
    showSignificance: !!ctx.state.showSignificance,
    series: { isCategory: true },
    categoriesOrder: ctx.groups,
    catLabelFn: ctx.groupLabel,
  });
}

function plotGrid(container, perRow) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:12px';
  container.appendChild(wrap);
  return (cellStyle = '') => {
    const cell = document.createElement('div');
    cell.style.cssText =
      `flex:0 0 calc(${100 / perRow}% - 12px);min-width:300px;box-sizing:border-box;${cellStyle}`;
    wrap.appendChild(cell);
    return cell;
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
  id: 'cellsketch-instance-morphology',
  label: 'Instance Morphology',
  group: 'Dataset Stats',

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
    const [row] = await ctx.queryRows(
      `SELECT COUNT(*) AS n, COUNT(DISTINCT ${ctx.sql.q('instance_entity')}) AS entities
       FROM ${unnestedSource(ctx, ['instance_entity', 'instance_volume_um3'])}`
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

    let entity = entities[0];
    const draw = async () => {
      plots.replaceChildren();
      const source = unnestedSource(ctx, ['instance_entity', ...metrics]);
      const where = `WHERE ${ctx.sql.q('instance_entity')} = ${esc(entity)}`;
      const [{ n }] = await ctx.queryRows(`SELECT COUNT(*) AS n FROM ${source} ${where}`);
      note(plots, `One point per ${entity} instance; n=${Number(n).toLocaleString()}.`);

      const nextCell = plotGrid(plots, ctx.groups.length <= 2 ? 3 : 2);
      for (const metric of metrics) {
        await drawDistribution(ctx, nextCell(), {
          numCol: metric,
          source,
          where,
          yLabel: labelFor(metric),
          title: labelFor(metric),
        });
      }
    };

    controls.appendChild(
      selector('Structure', entities, entity, (v) => {
        entity = v;
        draw();
      })
    );
    await draw();
  },
};

// ── Distances between structures ──────────────────────────────────────────────

const instanceDistances = {
  id: 'cellsketch-distances',
  label: 'Distances Between Structures',
  group: 'Dataset Stats',

  requires(schema) {
    return schema.allCols.includes('distance_um');
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
       FROM pp_data ${ctx.sql.andWhere(ctx.where, CELL_ROW)} ORDER BY e, t`
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
    let entity = entities[0];

    const draw = async () => {
      plots.replaceChildren();
      controls.replaceChildren();
      const targets = pairs.filter((p) => p.e === entity).map((p) => p.t);
      controls.appendChild(
        selector('Structure', entities, entity, (v) => {
          entity = v;
          draw();
        })
      );
      note(
        plots,
        `Smallest distance from each ${entity} instance to the named structure. ` +
          'Measured voxel centre to voxel centre, so 0 means they overlap.'
      );

      const source = unnestedSource(ctx, ['distance_entity', 'distance_target', 'distance_um']);
      const nextCell = plotGrid(plots, targets.length <= 1 ? 1 : 2);
      for (const target of targets) {
        const where =
          `WHERE ${ctx.sql.q('distance_entity')} = ${esc(entity)}` +
          ` AND ${ctx.sql.q('distance_target')} = ${esc(target)}`;
        await drawDistribution(ctx, nextCell(), {
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
      ON a."cell_id" = b."cell_id"
     AND a."distance_entity" = b."distance_entity"
     AND a."distance_label" = b."distance_label"
     AND a."distance_target" < b."distance_target"
    WHERE a."distance_entity" = ${esc(entity)}
  )`;
}

const instanceReach = {
  id: 'cellsketch-reach',
  label: 'Reaching Two Structures At Once',
  group: 'Dataset Stats',

  requires(schema) {
    return schema.allCols.includes('distance_um');
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

    let entity = entities[0];
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

      note(
        plots,
        `For every ${entity} instance, the larger of its two distances — the distance at ` +
          'which it is close to both structures. The curve is the share of instances at or ' +
          'below each value, so one that climbs early and steeply means most sit against both.'
      );

      const asNumbers = (q) => Array.from(q ?? [], Number);
      // A shared X range makes every panel directly comparable.
      const xMax = Math.max(...curves.flatMap((r) => asNumbers(r.quantiles)));
      const pairs = [...new Set(curves.map((r) => `${r.target_a} ${r.target_b}`))];
      const nextCell = plotGrid(plots, pairs.length <= 1 ? 1 : 2);

      for (const pair of pairs) {
        const [a, b] = pair.split(' ');
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
        ctx.plot.append(nextCell(), traces, {
          title: { text: `${a} & ${b}` },
          xaxis: { title: 'reaches both within (µm)', rangemode: 'tozero', range: [0, xMax] },
          yaxis: { title: `% of ${entity} instances`, ticksuffix: '%', range: [0, 102] },
          showlegend: true,
          legend: { orientation: 'h', y: -0.25 },
        });
      }
    };

    controls.appendChild(
      selector('Structure', entities, entity, (v) => {
        entity = v;
        draw();
      })
    );
    await draw();
  },
};

export default [instanceMorphology, instanceDistances, instanceReach];
