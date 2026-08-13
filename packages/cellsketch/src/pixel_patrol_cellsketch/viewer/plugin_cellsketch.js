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
  return `(SELECT ${ctx.sql.groupCol()} AS ${GROUP_ALIAS}, ${selects.join(', ')} FROM pp_data ${where})`;
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

// ── Instance morphology ───────────────────────────────────────────────────────

const instanceMorphology = {
  id: 'cellsketch-instance-morphology',
  label: 'Instance Morphology',
  group: 'Dataset Stats',

  requires(schema) {
    return schema.allCols.includes('instance_entity');
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

export default [instanceMorphology, instanceDistances];
