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


// ── Contacts & groups ─────────────────────────────────────────────────────────

// Enough slider steps to feel continuous without re-querying on every pixel.
const GAP_STEPS = 40;
const MAX_EDGES = 400000;

/** Every recorded contact, one row per instance pair. */
function contactEdgeSource(ctx) {
  return unnestedSource(ctx, [
    'contact_entity_a', 'contact_label_a', 'contact_entity_b', 'contact_label_b',
    'contact_gap_um',
  ]);
}

const pairLabel = (a, b) => [a, b].sort().join(' + ');

/**
 * Connected components of instances linked by contacts: a "contact group".
 *
 * Instances are seeded as singletons from the instance table, so the denominator is every
 * instance that could have touched, not only those that did. Identity is cell + entity +
 * label, since a label id alone repeats across cells and structures - and the cell each
 * group belongs to is tracked in its own map rather than parsed back out of a joined key,
 * because cell folder names may contain anything.
 */
export function contactGroups(instances, edges) {
  const parent = new Map();
  const cellOf = new Map();
  const find = (x) => {
    while (parent.get(x) !== x) {
      parent.set(x, parent.get(parent.get(x)));
      x = parent.get(x);
    }
    return x;
  };
  const ensure = (cell, entity, label) => {
    const k = JSON.stringify([cell, entity, label]);
    if (!parent.has(k)) {
      parent.set(k, k);
      cellOf.set(k, cell);
    }
    return k;
  };

  for (const i of instances) ensure(i.cell_id, i.entity, i.label);
  for (const e of edges) {
    const a = ensure(e.cell_id, e.entity_a, e.label_a);
    const b = ensure(e.cell_id, e.entity_b, e.label_b);
    parent.set(find(a), find(b));
  }

  const sizes = new Map();
  const rootCell = new Map();
  for (const k of parent.keys()) {
    const root = find(k);
    sizes.set(root, (sizes.get(root) ?? 0) + 1);
    if (!rootCell.has(root)) rootCell.set(root, cellOf.get(k));
  }
  return { sizes, rootCell, total: parent.size };
}

/** Group sizes and touch fraction per facet, from the facet each cell belongs to. */
export function summariseByFacet(groups, facetOfCell) {
  const byFacet = new Map();
  for (const [root, size] of groups.sizes) {
    const facet = facetOfCell.get(groups.rootCell.get(root)) ?? '(none)';
    const entry = byFacet.get(facet) ?? { sizes: [], instances: 0, touching: 0 };
    entry.instances += size;
    if (size > 1) {
      entry.sizes.push(size);
      entry.touching += size;
    }
    byFacet.set(facet, entry);
  }
  return byFacet;
}

const contactsAndGroups = {
  id: 'cellsketch-contacts',
  label: 'Contacts & Groups',
  group: 'Dataset Stats',

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

    // Which facet each cell belongs to: the active grouping, read off the cell rows.
    const facetRows = await ctx.queryRows(
      `SELECT ${ctx.sql.q('cell_id')} AS cell_id, ${ctx.sql.groupCol()} AS facet
       FROM pp_data ${ctx.sql.andWhere(ctx.where, CELL_ROW)}`
    );
    const facetOfCell = new Map(facetRows.map((r) => [String(r.cell_id), r.facet]));

    // Every instance, so a structure that touches nothing still counts in the denominator.
    const instances = (await ctx.queryRows(
      `SELECT ${ctx.sql.q('cell_id')} AS cell_id, ${ctx.sql.q('instance_entity')} AS entity,
              ${ctx.sql.q('instance_label')} AS label
       FROM ${unnestedSource(ctx, ['instance_entity', 'instance_label'])}`
    )).map((r) => ({ cell_id: String(r.cell_id), entity: r.entity, label: r.label }));

    const pairs = await ctx.queryRows(
      `SELECT DISTINCT ${ctx.sql.q('contact_entity_a')} AS a, ${ctx.sql.q('contact_entity_b')} AS b
       FROM ${edgeSource} ORDER BY 1, 2`
    );
    const pairOptions = [
      'all contacts', 'same-type contacts',
      ...new Set(pairs.map((p) => pairLabel(p.a, p.b))),
    ];

    const controls = document.createElement('div');
    controls.style.cssText =
      'display:flex;align-items:center;flex-wrap:wrap;gap:1rem;margin-bottom:.5rem';
    container.appendChild(controls);
    const plots = document.createElement('div');
    container.appendChild(plots);

    let pairChoice = pairOptions[0];
    let gapUm = widest;

    const draw = async () => {
      plots.replaceChildren();
      const conditions = [`${ctx.sql.q('contact_gap_um')} <= ${gapUm}`];
      if (pairChoice === 'same-type contacts') {
        conditions.push(`${ctx.sql.q('contact_entity_a')} = ${ctx.sql.q('contact_entity_b')}`);
      } else if (pairChoice !== 'all contacts') {
        const [a, b] = pairChoice.split(' + ');
        const ea = ctx.sql.q('contact_entity_a');
        const eb = ctx.sql.q('contact_entity_b');
        conditions.push(
          `((${ea} = ${esc(a)} AND ${eb} = ${esc(b)}) OR (${ea} = ${esc(b)} AND ${eb} = ${esc(a)}))`
        );
      }
      const edges = await ctx.queryRows(
        `SELECT ${ctx.sql.q('cell_id')} AS cell_id,
                ${ctx.sql.q('contact_entity_a')} AS entity_a,
                ${ctx.sql.q('contact_label_a')} AS label_a,
                ${ctx.sql.q('contact_entity_b')} AS entity_b,
                ${ctx.sql.q('contact_label_b')} AS label_b
         FROM ${edgeSource} WHERE ${conditions.join(' AND ')} LIMIT ${MAX_EDGES}`
      );
      if (edges.length >= MAX_EDGES) {
        ctx.plot.prependWarning(
          plots,
          `Only the first ${MAX_EDGES.toLocaleString()} contacts were grouped - lower the gap.`
        );
      }

      const groups = contactGroups(
        instances,
        edges.map((e) => ({ ...e, cell_id: String(e.cell_id) }))
      );
      const byFacet = summariseByFacet(groups, facetOfCell);
      const present = ctx.groups.filter((g) => byFacet.has(g));
      const ordered = present.length ? present : [...byFacet.keys()];

      note(
        plots,
        `A contact group is a cluster of instances chained together by contacts of `
        + `${gapUm.toFixed(3)} µm or less. An instance touches something exactly when it `
        + `lands in a group of two or more, so the two charts always agree.`
      );

      const nextCell = plotGrid(plots, 2);
      // Left: how many instances chain together.
      ctx.plot.append(
        nextCell(),
        ordered.map((facet) => ({
          type: 'box',
          name: ctx.groupLabel(facet),
          y: byFacet.get(facet).sizes,
          boxpoints: 'outliers',
          marker: { color: ctx.color.group(facet) },
        })),
        {
          title: { text: 'Contact group size' },
          yaxis: { title: 'instances per group', rangemode: 'tozero' },
          xaxis: { type: 'category' },
          showlegend: false,
        }
      );
      // Right: the share of instances that touch anything at all.
      ctx.plot.append(
        nextCell(),
        [{
          type: 'bar',
          x: ordered.map((f) => ctx.groupLabel(f)),
          y: ordered.map((f) => {
            const e = byFacet.get(f);
            return e.instances ? (e.touching / e.instances) * 100 : 0;
          }),
          marker: { color: ordered.map((f) => ctx.color.group(f)) },
          hovertemplate: '%{x}: %{y:.1f}% of instances touch<extra></extra>',
        }],
        {
          title: { text: 'Instances touching' },
          yaxis: { title: '% of instances', ticksuffix: '%', rangemode: 'tozero' },
          xaxis: { type: 'category' },
          showlegend: false,
        }
      );

      ctx.plot.statTable(plots, {
        headers: ['', 'instances', 'groups', 'largest', 'mean size', 'touching'],
        rows: ordered.map((facet) => {
          const e = byFacet.get(facet);
          const mean = e.sizes.length ? e.sizes.reduce((a, b) => a + b, 0) / e.sizes.length : 0;
          return [
            ctx.groupLabel(facet),
            e.instances.toLocaleString(),
            e.sizes.length.toLocaleString(),
            e.sizes.length ? Math.max(...e.sizes) : 0,
            mean.toFixed(2),
            e.instances ? `${((e.touching / e.instances) * 100).toFixed(1)}%` : '-',
          ];
        }),
      });
    };

    controls.appendChild(selector('Contacts', pairOptions, pairChoice, (value) => {
      pairChoice = value;
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
