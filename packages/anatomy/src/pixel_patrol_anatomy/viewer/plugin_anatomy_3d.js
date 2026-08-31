/**
 * Anatomy 3D widgets: the object, and the instances worth looking at.
 *
 * Geometry lives beside the report, one geometry.parquet per object. Meshes would multiply
 * the size of a table every stats query loads (36 MB for one real object against a 1 MB
 * report). The object row carries the path in `mesh_geometry_file`, and these widgets query
 * that file through the viewer's own DuckDB connection, so asking for the twelve roundest
 * granules costs twelve meshes rather than an object's worth.
 *
 * Being in the viewer means the filter, grouping and palette are the ones every other widget
 * uses: "the outliers in that violin" and "these twelve meshes" are the same instances.
 *
 * Self-contained: a static build inlines each plugin as its own data: URL, so cross-file
 * imports would not resolve. three.js is the exception, imported lazily from a sibling file
 * if there is one and from a CDN otherwise.
 */

const OBJECT_ROW = '"obs_level" = 0';
const THREE_VERSION = '0.160.0';
const CDN_THREE = `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/build/three.module.js`;

// Instance meshes per scene before some are left out. An object has ~8k; merged they cost
// one draw call but real memory.
const MAX_MESHES = 4000;
// How many rows of thumbnails to draw, rather than how many thumbnails: the grid fits as many
// columns as the card is wide, so a fixed count leaves the last row half empty at most widths.
const GALLERY_ROWS = [2, 4, 8];
// Whatever the width, this many thumbnails is enough: each is its own WebGL render.
const MAX_THUMBS = 120;
// The gallery's group filter: every group, or one of them.
const ALL_GROUPS = '__all__';
const THUMB_PX = 104;

const METRIC_LABELS = {
  volume_um3: 'Volume (µm³)',
  area_um2: 'Area (µm²)',
  surface_area_um2: 'Surface area (µm²)',
  perimeter_um: 'Perimeter (µm)',
  sphericity: 'Sphericity',
  circularity: 'Circularity',
  aspect_ratio_major_minor: 'Aspect ratio (major/minor)',
  branches: 'Skeleton branches',
  length_um: 'Skeleton length (µm)',
  tortuosity: 'Tortuosity',
  distance_to_closest_same_type_um: 'Distance to nearest same type (µm)',
  polar_dist_um: 'Distance from object centre (µm)',
  polar_az_deg: 'Azimuth from object centre (°)',
  polar_el_deg: 'Elevation from object centre (°)',
  polar_angle_deg: 'Angle from object centre (°)',
};
const METRIC_ORDER = Object.keys(METRIC_LABELS);

// Metrics only one dimensionality has. A geometry file declares every column whether it was
// filled or not, so the switch reads the report instead: a 2D batch has total_area_um2 and
// no total_volume_um3.
const METRICS_3D_ONLY = new Set(['volume_um3', 'surface_area_um2', 'sphericity',
                                 'polar_az_deg', 'polar_el_deg']);
const METRICS_2D_ONLY = new Set(['area_um2', 'perimeter_um', 'circularity',
                                 'polar_angle_deg']);

/** True when this report's objects are planes rather than volumes. */
function isPlanarReport(schema) {
  return !schema.allCols.includes('total_volume_um3')
    && schema.allCols.includes('total_area_um2');
}

/** The metrics worth offering for this report, in curated order. */
function metricsFor(schema) {
  const planar = isPlanarReport(schema);
  const drop = planar ? METRICS_3D_ONLY : METRICS_2D_ONLY;
  return METRIC_ORDER.filter((m) => !drop.has(m));
}

/**
 * One row's drawable geometry: a triangle mesh for a volume, closed outline loops for a
 * plane. Same container, two indices per element instead of three, so the caller only needs
 * `flat` to decide between a Mesh and LineSegments.
 */
function decodeDrawable(row) {
  if (row.mesh) {
    const geometry = decodePayload(row.mesh, 3);
    if (geometry) return { geometry, flat: false };
  }
  if (row.outline) {
    const geometry = decodePayload(row.outline, 2);
    if (geometry) return { geometry, flat: true };
  }
  return null;
}

// A row has a mesh or an outline; a mixed batch has both kinds of row.
const HAS_GEOMETRY = '("mesh" IS NOT NULL OR "outline" IS NOT NULL)';
// The size to rank by, whichever the object has. Both columns exist in every geometry file.
const GEOMETRY_SIZE = 'COALESCE("volume_um3", "area_um2")';

// Entity colours, so a structure keeps its colour across both widgets and every object.
// Tableau 10 rather than a muted chart palette: the same ten hues in the same order, but
// saturated enough to survive being multiplied by a light and read across a page.
const ENTITY_COLOURS = [
  '#33ddff', '#ff8f00', '#acff6c', '#ff2728', '#9467ff',
  '#f1a69b', '#e377ff', '#7f7f7f', '#ccff22', '#17beff',
];

const esc = (v) => `'${String(v).replace(/'/g, "''")}'`;
const num = (v) => (v === null || v === undefined ? null : Number(v));
const labelFor = (col) => METRIC_LABELS[col] ?? col;

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

// ── three.js, loaded once and only when a 3D widget is actually opened ─────────

let _threePromise = null;

/**
 * three.js as an ES module. A copy next to this plugin wins (offline installs); otherwise
 * the CDN, as the viewer's own DuckDB does in a light static build.
 */
function loadThree() {
  if (_threePromise) return _threePromise;
  _threePromise = (async () => {
    try {
      // import.meta.url is a data: URL in a single-file static build, where there is no
      // sibling to resolve against and this throws - the CDN is the answer there anyway.
      const sibling = new URL('./three.module.js', import.meta.url).href;
      const probe = await fetch(sibling, { method: 'HEAD' });
      if (probe.ok) return await import(/* @vite-ignore */ sibling);
    } catch { /* no local copy: fall through to the CDN */ }
    return import(/* @vite-ignore */ CDN_THREE);
  })().catch((err) => {
    _threePromise = null;
    throw err;
  });
  return _threePromise;
}

// ── payloads ──────────────────────────────────────────────────────────────────

/**
 * Decode one geometry blob into vertex positions and indices.
 *
 * The container is written by mesh.py and shared with geometry_to_blender.py:
 *   [uint32 nV][uint32 nI][float32x3 min_xyz][float32x3 scale_xyz]
 *   [uint16 x nV*3 quantised XYZ][uint32 x nI*perIndex indices]
 * Vertices are µm in XYZ, dequantised against the min/scale that follow the header.
 */
function decodePayload(bytes, perIndex = 3) {
  if (!bytes || bytes.byteLength < 32) return null;
  // Copy out of the Arrow buffer: a Uint16Array view needs its own alignment, and the
  // blob sits at an arbitrary offset inside the batch's memory.
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const header = new Uint32Array(buffer, 0, 2);
  const nVerts = header[0];
  const nIndices = header[1];
  if (!nVerts || !nIndices) return null;
  const params = new Float32Array(buffer, 8, 6);
  const quantised = new Uint16Array(buffer, 32, nVerts * 3);
  const indices = new Uint32Array(buffer.slice(32 + nVerts * 6));

  const positions = new Float32Array(nVerts * 3);
  for (let i = 0; i < nVerts; i += 1) {
    positions[i * 3] = params[0] + (quantised[i * 3] / 65535) * params[3];
    positions[i * 3 + 1] = params[1] + (quantised[i * 3 + 1] / 65535) * params[4];
    positions[i * 3 + 2] = params[2] + (quantised[i * 3 + 2] / 65535) * params[5];
  }
  return { positions, indices: indices.subarray(0, nIndices * perIndex) };
}

// ── colour ────────────────────────────────────────────────────────────────────

function entityColour(name, entities) {
  const index = Math.max(0, entities.indexOf(name));
  return ENTITY_COLOURS[index % ENTITY_COLOURS.length];
}

/** #rrggbb → [r, g, b] in 0..1, the form a vertex-colour attribute wants. */
function toRgb(hex) {
  const value = String(hex).replace('#', '');
  const full = value.length === 3 ? value.split('').map((c) => c + c).join('') : value;
  const int = parseInt(full, 16);
  return [((int >> 16) & 255) / 255, ((int >> 8) & 255) / 255, (int & 255) / 255];
}

/** Viridis, sampled - the same ramp the standalone viewer used for metric colouring. */
const VIRIDIS = ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c',
  '#28ae80', '#5ec962', '#addc30', '#fde725'];

function viridisAt(t) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const scaled = clamped * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = toRgb(VIRIDIS[i]);
  const b = toRgb(VIRIDIS[i + 1]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ── contact groups ────────────────────────────────────────────────────────────

/**
 * Union-find over one object's contact edges: which instances chain together at this gap.
 * Every drawn instance is seeded as its own group, so something touching nothing is a
 * singleton rather than absent, which is the rule Contacts & Groups counts by.
 */
function contactGroups(instances, edges) {
  const parent = new Map();
  const find = (key) => {
    while (parent.get(key) !== key) {
      parent.set(key, parent.get(parent.get(key)));
      key = parent.get(key);
    }
    return key;
  };
  const union = (a, b) => {
    const [ra, rb] = [find(a), find(b)];
    if (ra !== rb) parent.set(ra, rb);
  };
  for (const key of instances) parent.set(key, key);
  for (const edge of edges) {
    if (parent.has(edge.a) && parent.has(edge.b)) union(edge.a, edge.b);
  }
  const members = new Map();
  for (const key of instances) {
    const root = find(key);
    members.set(root, (members.get(root) ?? 0) + 1);
  }
  return { rootOf: (key) => find(key), sizeOfRoot: (root) => members.get(root) ?? 1 };
}

// ── scene ─────────────────────────────────────────────────────────────────────

/**
 * Where an instance goes when the explode slider is up: out along its own direction from the
 * object centre, by its own distance from it, so a crowded interior opens up without anything
 * crossing anything else.
 *
 * `polar.n` is read as [x, y, z] from polar_nx/ny/nz, which is the order the meshes are in,
 * so nothing is reordered here. Reordering it to ZYX mirrored the offset across x and z and
 * sent every instance somewhere it had never been.
 */
function explodeOffset(polar, factor) {
  const dist = polar?.dist;
  if (!factor || !dist) return [0, 0, 0];
  const [nx, ny, nz] = polar.n;
  return [nx * dist * factor, ny * dist * factor, nz * dist * factor];
}

// Near-white, and the light tuning it needs: on white the ambient term has to stay down and the
// ground tint up, or everything turns to pale mush. The intensities sum to about one on the
// brightest face, past which a lit surface clips towards white and takes its colour with it.
const STAGE = {
  background: 0xf7f7fa, ambient: 0.22, sky: 0xffffff, ground: 0x9c9caa, hemisphere: 0.38,
  key: 0.85, fill: 0.14, rim: 0.1,
};

// How far the key light stands off the camera. Fixed in the world it would leave whole sides
// unlit, so turning the back to the front would show a dark shape; riding with the camera at an
// offset, the surface being looked at is always lit while the light still moves relative to the
// object, so the shadows travel across it as you orbit. Head-on would flatten everything, and
// much further round the near side starts falling into its own shadow.
const LIGHT_OFFSET = (35 * Math.PI) / 180;

/** A renderer, scene and camera for one canvas, plus the orbit interaction. */
function makeStage(THREE, canvas, { shadows = true } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  // Self-shadowing is what makes a pile of blobs read as a pile: nothing else tells you which
  // granule is in front of which. It is one extra pass over the same merged geometry, so the
  // gallery, which draws one instance per thumbnail hundreds of times, leaves it off.
  renderer.shadowMap.enabled = shadows;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1e6);
  const target = new THREE.Vector3();
  const spherical = { radius: 5, theta: Math.PI * 0.25, phi: Math.PI * 0.35 };
  let framed = null;                 // { centre, size } of what is drawn, from frameShadow()

  // Depth comes from the light falling off, so the ambient term stays low and the rest is
  // directional: a key from above front, a weak fill so the shadow side is not black, and a rim
  // from behind to lift each surface off the one behind it. The hemisphere light tints up-facing
  // surfaces light and down-facing ones dark, which shades the pile as a whole.
  const ambient = new THREE.AmbientLight(0xffffff, STAGE.ambient);
  const hemisphere = new THREE.HemisphereLight(STAGE.sky, STAGE.ground, STAGE.hemisphere);
  const key = new THREE.DirectionalLight(0xffffff, STAGE.key);
  const keyAt = new THREE.Object3D();      // a directional light shadows towards its target
  key.castShadow = shadows;
  key.shadow.mapSize.set(1024, 1024);
  key.target = keyAt;
  const fill = new THREE.DirectionalLight(0xffffff, STAGE.fill);
  fill.position.set(-1, -0.5, 0.5);
  const rim = new THREE.DirectionalLight(0xffffff, STAGE.rim);
  rim.position.set(-0.6, 0.4, -1.6);
  scene.add(ambient, hemisphere, key, keyAt, fill, rim);
  scene.background = new THREE.Color(STAGE.background);

  /** Stand the key light off the camera, up and to one side, pointed at the object. */
  const placeKey = () => {
    if (!framed) return;
    const { centre, size } = framed;
    const theta = spherical.theta + LIGHT_OFFSET;
    // Above the camera, but never straight overhead: a raking light is what shows the form.
    const phi = Math.max(0.25, spherical.phi - 0.45);
    const sinPhi = Math.sin(phi);
    key.position.set(
      centre.x + size * 2 * sinPhi * Math.sin(theta),
      centre.y + size * 2 * Math.cos(phi),
      centre.z + size * 2 * sinPhi * Math.cos(theta),
    );
    keyAt.position.copy(centre);
  };

  /**
   * Measure what is drawn and fit the key light's shadow frustum to it.
   *
   * Called on every repaint, not only when the camera reframes: turning a structure on grows the
   * scene, and a stale frustum clips the shadow off it. Bias is in the shadow camera's own depth
   * units, so it does not scale with the object; normalBias is in µm, so it does.
   */
  const frameShadow = () => {
    const box = new THREE.Box3();
    for (const child of scene.children) {
      if (child.isMesh || child.isLineSegments) box.expandByObject(child);
    }
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    framed = { centre: box.getCenter(new THREE.Vector3()), size };
    placeKey();
    const shadow = key.shadow.camera;
    shadow.left = -size * 0.75;
    shadow.right = size * 0.75;
    shadow.top = size * 0.75;
    shadow.bottom = -size * 0.75;
    shadow.near = size * 0.05;
    shadow.far = size * 6;
    shadow.updateProjectionMatrix();
    key.shadow.bias = -0.0008;               // acne otherwise, at grazing angles
    if ('normalBias' in key.shadow) key.shadow.normalBias = size * 0.003;
  };

  const apply = () => {
    const sinPhi = Math.sin(spherical.phi);
    camera.position.set(
      target.x + spherical.radius * sinPhi * Math.sin(spherical.theta),
      target.y + spherical.radius * Math.cos(spherical.phi),
      target.z + spherical.radius * sinPhi * Math.cos(spherical.theta),
    );
    camera.lookAt(target);
    placeKey();                              // the light rides with the camera
    renderer.render(scene, camera);
  };

  const stage = {
    THREE,
    scene,
    camera,
    renderer,
    draw: apply,
    frameShadow,
    /** Frame the geometry. The lights are in the scene too, and sit outside it. */
    fit() {
      frameShadow();
      if (!framed) return;
      target.copy(framed.centre);
      spherical.radius = framed.size * 1.1;
      camera.near = framed.size / 1000;
      camera.far = framed.size * 100;
      camera.updateProjectionMatrix();
      // Fog over the depth of the object itself, so the far side fades into the background. It
      // starts behind the near face, so nothing in front of the middle is dulled.
      scene.fog = new THREE.Fog(STAGE.background, spherical.radius * 0.85, spherical.radius * 2.8);
      apply();
    },
    resize(width, height) {
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      apply();
    },
    /** Drop every object drawn so far, freeing its GPU buffers. */
    clear() {
      for (const child of [...scene.children]) {
        if (!child.isMesh && !child.isLineSegments) continue;
        child.geometry?.dispose();
        child.material?.dispose();
        scene.remove(child);
      }
    },
    dispose() {
      stage.clear();
      renderer.dispose();
      renderer.forceContextLoss?.();
    },
  };

  // Orbit, dolly and pan by hand: three's OrbitControls is a separate module that imports
  // three by bare specifier, which needs an import map the viewer page does not have.
  let dragging = null;
  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', (event) => {
    dragging = { x: event.clientX, y: event.clientY, pan: event.button === 2 || event.shiftKey };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const dx = event.clientX - dragging.x;
    const dy = event.clientY - dragging.y;
    dragging.x = event.clientX;
    dragging.y = event.clientY;
    if (dragging.pan) {
      camera.updateMatrixWorld();
      const e = camera.matrixWorld.elements;
      const scale = spherical.radius * 0.0015;
      target.x -= (e[0] * dx - e[4] * dy) * scale;
      target.y -= (e[1] * dx - e[5] * dy) * scale;
      target.z -= (e[2] * dx - e[6] * dy) * scale;
    } else {
      spherical.theta -= dx * 0.006;
      spherical.phi = Math.max(0.02, Math.min(Math.PI - 0.02, spherical.phi - dy * 0.006));
    }
    apply();
  });
  const stop = (event) => {
    if (!dragging) return;
    dragging = null;
    canvas.releasePointerCapture?.(event.pointerId);
  };
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    spherical.radius = Math.max(1e-4, spherical.radius * (1 + Math.sign(event.deltaY) * 0.12));
    apply();
  }, { passive: false });

  return stage;
}

/**
 * Merge decoded instances into one geometry with per-vertex colours: one draw call for the
 * whole object, however it is coloured, because the colour rides on the vertices.
 */

function mergeInstances(THREE, items) {
  let vertexCount = 0;
  let indexCount = 0;
  for (const item of items) {
    vertexCount += item.geometry.positions.length / 3;
    indexCount += item.geometry.indices.length;
  }
  if (!vertexCount) return null;

  const positions = new Float32Array(vertexCount * 3);
  const colours = new Float32Array(vertexCount * 3);
  const indices = vertexCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);
  let vertexOffset = 0;
  let indexOffset = 0;
  for (const item of items) {
    const { positions: src, indices: idx } = item.geometry;
    const [r, g, b] = item.rgb;
    const [ox, oy, oz] = item.offset ?? [0, 0, 0];
    for (let i = 0; i < src.length; i += 3) {
      const at = vertexOffset * 3 + i;
      positions[at] = src[i] + ox;
      positions[at + 1] = src[i + 1] + oy;
      positions[at + 2] = src[i + 2] + oz;
      colours[at] = r;
      colours[at + 1] = g;
      colours[at + 2] = b;
    }
    for (let i = 0; i < idx.length; i += 1) indices[indexOffset + i] = idx[i] + vertexOffset;
    vertexOffset += src.length / 3;
    indexOffset += idx.length;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

// ── where the geometry is ─────────────────────────────────────────────────────

const OVERRIDE_KEY = 'anatomy.geometryBase';

/**
 * The objects in scope that have geometry, and where it is.
 *
 * The recorded path is absolute and local, which is what DuckDB needs when it runs
 * natively behind `pixel-patrol view`. In a browser-only viewer that path means nothing,
 * so a base URL kept in localStorage can redirect the lookup at a hosted copy.
 */
async function geometryObjects(ctx) {
  const where = ctx.sql.andWhere(
    ctx.sql.andWhere(ctx.where, OBJECT_ROW),
    `${ctx.sql.q('mesh_geometry_file')} IS NOT NULL`
  );
  const rows = await ctx.queryRows(
    `SELECT ${ctx.sql.q('object_id')} AS id,
            ${ctx.sql.groupCol()} AS grp,
            ${ctx.sql.q('mesh_geometry_file')} AS path
     FROM pp_data ${where}`
  );
  const base = (window.localStorage?.getItem(OVERRIDE_KEY) || '').replace(/\/+$/, '');
  return rows
    .filter((r) => r.id !== null && r.id !== undefined)
    .map((r) => ({
      id: String(r.id),
      group: r.grp,
      path: base ? `${base}/${r.id}/geometry.parquet` : String(r.path),
    }));
}

function sourceOf(objects) {
  return `read_parquet([${objects.map((c) => esc(c.path)).join(', ')}])`;
}

/**
 * Every structure with geometry anywhere in scope, in one fixed order. Both widgets colour
 * by position in it, so a structure keeps its colour across objects; taken per object, one
 * missing structure would shift every other colour.
 */
async function entityPalette(ctx, objects) {
  const rows = await ctx.queryRows(
    `SELECT DISTINCT "entity_name" AS name FROM ${sourceOf(objects)}
     WHERE ${HAS_GEOMETRY} ORDER BY 1`
  );
  const order = rows.map((r) => String(r.name));
  const chosen = new Map();
  // A run can be given a settings file naming a colour per structure; it lands on the report's
  // entity rows, which is where the other widgets read it too, so one structure is one colour
  // everywhere. Older reports have no such column.
  if (ctx.schema.allCols.includes('entity_colour')) {
    const chose = await ctx.queryRows(
      `SELECT DISTINCT ${ctx.sql.q('entity_name')} AS name,
              ${ctx.sql.q('entity_colour')} AS colour
       FROM pp_all WHERE ${ctx.sql.q('entity_colour')} IS NOT NULL`
    );
    for (const row of chose) chosen.set(String(row.name), String(row.colour));
  }
  return { order, of: (name) => chosen.get(name) ?? entityColour(name, order) };
}

/** A row of controls; every widget here puts its selectors in one. */
function controlBar(container) {
  const bar = document.createElement('div');
  bar.style.cssText =
    'display:flex;align-items:center;flex-wrap:wrap;gap:.9rem;margin-bottom:.6rem';
  container.appendChild(bar);
  return bar;
}

function selector(labelText, values, current, onChange, { labels = null } = {}) {
  const wrap = document.createElement('label');
  wrap.style.cssText = 'display:inline-flex;align-items:center;gap:.4rem;font-size:.8rem';
  wrap.append(labelText);
  const select = document.createElement('select');
  select.style.cssText = 'font-size:.8rem;padding:.15rem .3rem;max-width:22rem';
  for (const value of values) {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = labels ? (labels.get(value) ?? String(value)) : String(value);
    if (String(value) === String(current)) option.selected = true;
    select.appendChild(option);
  }
  select.addEventListener('change', () => onChange(select.value));
  wrap.appendChild(select);
  return wrap;
}

function slider(labelText, { min, max, step, value, format }, onInput, onCommit) {
  const wrap = document.createElement('label');
  wrap.style.cssText = 'display:inline-flex;align-items:center;gap:.4rem;font-size:.8rem';
  const input = document.createElement('input');
  input.type = 'range';
  Object.assign(input, { min, max, step, value });
  input.style.width = '110px';
  const readout = document.createElement('span');
  readout.style.cssText = 'font-variant-numeric:tabular-nums;color:#555;min-width:3.4rem';
  readout.textContent = format(Number(value));
  input.addEventListener('input', () => {
    readout.textContent = format(Number(input.value));
    onInput?.(Number(input.value));
  });
  input.addEventListener('change', () => onCommit?.(Number(input.value)));
  wrap.append(labelText, input, readout);
  return wrap;
}

function note(container, text) {
  const p = document.createElement('p');
  p.style.cssText = 'font-size:.75rem;color:#666;margin:.35rem 0 0';
  p.textContent = text;
  container.appendChild(p);
  return p;
}

function emptyState(container, text) {
  const p = document.createElement('p');
  p.style.cssText = 'font-size:.85rem;color:#666';
  p.textContent = text;
  container.appendChild(p);
}

/** True for the failures that are about drawing rather than about the data. */
function isDisplayProblem(error) {
  const message = String(error?.message ?? error);
  return /WebGL|context|three|Failed to fetch dynamically imported module/i.test(message);
}

/**
 * Report a failure, and offer the way out of the one that has one.
 *
 * Drawing failures (no WebGL, three.js unreachable) have no data-side fix. Reading failures
 * usually do: the report records an absolute local path, which DuckDB-WASM cannot open, so
 * the widget takes a base URL for a hosted copy and remembers it.
 */
function sourceProblem(container, error, redraw) {
  const box = document.createElement('div');
  box.style.cssText =
    'border:1px solid #f0d0c8;background:#fdf3f1;border-radius:6px;padding:.6rem .8rem;'
    + 'font-size:.8rem;color:#7a2e22;margin-bottom:.6rem';
  const display = isDisplayProblem(error);
  const headline = document.createElement('p');
  headline.style.margin = '0 0 .4rem';
  headline.textContent = display
    ? 'Nothing could be drawn: this browser has no working WebGL, or three.js could not be '
      + 'loaded. Everything else in the report is unaffected.'
    : window.__PP_SERVER
      ? 'The geometry file could not be read.'
      : 'The geometry file could not be read. A browser-only viewer cannot open the local '
        + 'path the report recorded. Serve the report with `pixel-patrol view`, or give the '
        + 'URL of a hosted copy of the mesh folder.';
  box.appendChild(headline);

  // No input when there is nothing to redirect: a missing WebGL context is not a path.
  const row = document.createElement('div');
  row.style.cssText =
    `display:${display ? 'none' : 'flex'};gap:.4rem;align-items:center;flex-wrap:wrap`;
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'https://host/report_meshes';
  input.value = window.localStorage?.getItem(OVERRIDE_KEY) || '';
  input.style.cssText = 'flex:1 1 22rem;font-size:.78rem;padding:.2rem .35rem';
  const apply = document.createElement('button');
  apply.textContent = 'Use this folder';
  apply.style.cssText = 'font-size:.78rem;padding:.2rem .5rem';
  apply.addEventListener('click', () => {
    try { window.localStorage?.setItem(OVERRIDE_KEY, input.value.trim()); } catch { /* private mode */ }
    redraw();
  });
  row.append(input, apply);
  box.appendChild(row);

  const detail = document.createElement('p');
  detail.style.cssText = 'margin:.4rem 0 0;font-size:.72rem;opacity:.8';
  detail.textContent = String(error?.message ?? error);
  box.appendChild(detail);
  container.appendChild(box);
}

/** A canvas that fills its container and keeps the stage sized to it. */
function stageCanvas(container, height) {
  const holder = document.createElement('div');
  holder.style.cssText = `position:relative;width:100%;height:${height}px;border-radius:6px;overflow:hidden`;
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;height:100%;display:block';
  holder.appendChild(canvas);
  container.appendChild(holder);
  return { holder, canvas };
}

function busy(holder, text) {
  const overlay = document.createElement('div');
  overlay.style.cssText =
    'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
    + 'background:rgba(255,255,255,.75);font-size:.8rem;color:#333;z-index:2';
  overlay.textContent = text;
  holder.appendChild(overlay);
  return () => overlay.remove();
}

// ── Object 3D ───────────────────────────────────────────────────────────────────

// The object view's stage, kept at module scope so a re-render hands the same WebGL context
// on rather than opening another one the browser will later drop.
let _objectStage = null;

const object3d = {
  id: 'anatomy-object-3d',
  label: 'Object in 3D',
  group: 'Visualization',
  info: [
    'One object, as it was segmented. **Drag** to orbit, **scroll** to zoom, **shift-drag** to '
    + 'pan.',
    '',
    '- Colour **by structure** to see the layout',
    '- **by a metric** to find where the extremes sit',
    '- **by contact group** to see which instances chain together at a given gap',
    '',
    '**Explode** pushes each instance out along its own direction from the object centre, which '
    + 'opens up a crowded interior without moving anything across it.',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.includes('mesh_geometry_file');
  },

  async overviewMessage(ctx) {
    const objects = await geometryObjects(ctx);
    if (!objects.length) {
      return { text: 'No geometry for the objects in view. Re-run with <b>--with-mesh</b>.',
        warning: true };
    }
    return `<strong>${objects.length}</strong> object(s) can be opened in 3D, `
      + 'every structure in place.';
  },

  async overviewPlot(container, ctx) {
    const objects = await geometryObjects(ctx);
    if (!objects.length) return false;
    const one = objects.slice(0, 1);
    // The whole-structure masks, if they are small enough to be worth a tile: a real object
    // mask can be 100k vertices and megabytes. Otherwise the biggest instances that do fit,
    // which still reads as one object full of structures.
    const masks = await previewGeometry(ctx, one,
                                       { rowType: 'file', limit: 3, maxVertices: 40_000 });
    const shown = masks.length ? masks
      : await previewGeometry(ctx, one, { rowType: 'instance', limit: 30,
                                         maxVertices: 25_000, perEntity: 8 });
    if (!shown.length) return false;
    return paintScene(container, shown, await entityPalette(ctx, objects));
  },

  async render(container, ctx) {
    const objects = await geometryObjects(ctx);
    if (!objects.length) {
      emptyState(container,
        'No geometry beside this report. Re-run with --with-mesh (or `anatomy mesh`) to '
        + 'write one geometry.parquet per object.');
      return;
    }

    const bar = controlBar(container);
    const { holder, canvas } = stageCanvas(container, 520);
    const footer = note(container, '');

    const shortOf = new Map(objects.map((c) => [c.id, c.id.length > 34 ? `…${c.id.slice(-32)}` : c.id]));
    // A 2D batch has no volume to rank or colour by, so offer only what the report holds.
    const metricsAvailable = metricsFor(ctx.schema);
    const state = {
      object: objects[0],
      colourBy: 'entity',
      metric: metricsAvailable[0],
      entities: null,          // null until the first read; then a Set of enabled names
      palette: null,           // the cohort-wide structure colours, read once
      explode: 0,
      gap: null,
      maxGap: null,
    };
    let loaded = null;         // decoded rows for the current object + entity choice
    // One context, reused across renders: a browser drops the oldest once a page holds more
    // than a handful, and every filter change re-renders every widget.
    _objectStage?.dispose();
    _objectStage = null;

    const redraw = () => {
      container.replaceChildren();
      object3d.render(container, ctx);
    };

    /**
     * What is in this object's geometry file, and how big: the cheap query. One entity can
     * contribute both row kinds (a whole-structure mask and a mesh per instance), so counts
     * are folded into one entry per name, which is what the user toggles.
     */
    const readSummary = async () => {
      const rows = await ctx.queryRows(
        `SELECT "entity_name" AS name, "row_type" AS row_type,
                COUNT(*) AS n, SUM("mesh_faces") AS faces
         FROM ${sourceOf([state.object])}
         WHERE ${HAS_GEOMETRY} GROUP BY 1, 2 ORDER BY 1`
      );
      const [gaps] = await ctx.queryRows(
        `SELECT MAX("gap_um") AS widest FROM ${sourceOf([state.object])} WHERE "row_type" = 'contact'`
      );
      const byName = new Map();
      for (const row of rows) {
        const name = String(row.name);
        const entry = byName.get(name)
          ?? { name, instances: 0, masks: 0, faces: 0 };
        if (row.row_type === 'file') entry.masks += Number(row.n);
        else entry.instances += Number(row.n);
        entry.faces += Number(row.faces ?? 0);
        byName.set(name, entry);
      }
      return { entities: [...byName.values()], widestGap: num(gaps?.widest) };
    };

    const readGeometry = async (summary) => {
      const wanted = summary.entities
        .filter((e) => state.entities.has(e.name))
        .map((e) => esc(e.name));
      if (!wanted.length) return { instances: [], masks: [], edges: [] };

      const metrics = [...new Set([state.metric, 'polar_dist_um', 'polar_nx', 'polar_ny', 'polar_nz'])];
      const columns = metrics.map((m) => `"${m}"`).join(', ');
      const rows = await ctx.queryRows(
        `SELECT "entity_name" AS entity, "row_type" AS row_type, "label_id" AS label,
                ${columns}, "mesh" AS mesh, "outline" AS outline
         FROM ${sourceOf([state.object])}
         WHERE ${HAS_GEOMETRY} AND "entity_name" IN (${wanted.join(', ')})
         ORDER BY ${GEOMETRY_SIZE} DESC NULLS LAST
         LIMIT ${MAX_MESHES}`
      );
      const instances = [];
      const masks = [];
      for (const row of rows) {
        const drawable = decodeDrawable(row);
        if (!drawable) continue;
        const item = {
          entity: String(row.entity),
          label: row.label === null || row.label === undefined ? null : String(row.label),
          metric: num(row[state.metric]),
          polar: {
            dist: num(row.polar_dist_um) ?? 0,
            n: [num(row.polar_nx) ?? 0, num(row.polar_ny) ?? 0, num(row.polar_nz) ?? 0],
          },
          geometry: drawable.geometry,
          flat: drawable.flat,
        };
        (row.row_type === 'file' ? masks : instances).push(item);
      }

      let edges = [];
      if (state.colourBy === 'group' && state.gap !== null) {
        edges = (await ctx.queryRows(
          `SELECT "entity_a" AS ea, "label_a" AS la, "entity_b" AS eb, "label_b" AS lb
           FROM ${sourceOf([state.object])}
           WHERE "row_type" = 'contact' AND "gap_um" <= ${state.gap}`
        )).map((e) => ({ a: `${e.ea}|${e.la}`, b: `${e.eb}|${e.lb}` }));
      }
      return { instances, masks, edges };
    };

    /** Paint what is loaded, with the current colouring and explode factor. */
    const paint = () => {
      const stage = _objectStage;
      if (!stage || !loaded) return;
      const { THREE } = stage;
      stage.clear();

      // Colours come from the object's full entity list, not the drawn subset, so turning a
      // structure off does not recolour the ones left.
      let colourOf;
      if (state.colourBy === 'metric') {
        const values = loaded.instances.map((i) => i.metric).filter(Number.isFinite);
        const low = Math.min(...values);
        const high = Math.max(...values);
        const span = high - low || 1;
        colourOf = (item) => (Number.isFinite(item.metric)
          ? viridisAt((item.metric - low) / span)
          : [0.6, 0.6, 0.6]);
      } else if (state.colourBy === 'group') {
        const keys = loaded.instances.map((i) => `${i.entity}|${i.label}`);
        const groups = contactGroups(keys, loaded.edges);
        const clusterColour = new Map();
        colourOf = (item) => {
          const root = groups.rootOf(`${item.entity}|${item.label}`);
          // A singleton touches nothing: grey it, so the clusters are what stands out.
          if (groups.sizeOfRoot(root) < 2) return [0.72, 0.72, 0.75];
          if (!clusterColour.has(root)) {
            clusterColour.set(
              root, toRgb(ENTITY_COLOURS[clusterColour.size % ENTITY_COLOURS.length]));
          }
          return clusterColour.get(root);
        };
      } else {
        colourOf = (item) => toRgb(state.palette.of(item.entity));
      }

      const offsetOf = (item) => explodeOffset(item.polar, state.explode);

      // A plane has no surface to shade: same merge, built as LineSegments.
      const asItem = (item) => ({
        geometry: item.geometry, rgb: colourOf(item), offset: offsetOf(item),
      });
      const surfaces = mergeInstances(THREE, loaded.instances.filter((i) => !i.flat).map(asItem));
      if (surfaces) {
        const mesh = new THREE.Mesh(surfaces, new THREE.MeshPhongMaterial({
          vertexColors: true, side: THREE.DoubleSide, flatShading: false,
          shininess: 8, specular: 0x2a2a33,
        }));
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        stage.scene.add(mesh);
      }
      const outlines = mergeInstances(THREE, loaded.instances.filter((i) => i.flat).map(asItem));
      if (outlines) {
        stage.scene.add(new THREE.LineSegments(outlines, new THREE.LineBasicMaterial({
          vertexColors: true,
        })));
      }

      // Masks enclose the instances, so they are see-through and never merged: each wants
      // its own transparency. In a plane a boundary line does that job without any.
      for (const mask of loaded.masks) {
        const geometry = mergeInstances(THREE, [{ geometry: mask.geometry, rgb: colourOf(mask) }]);
        if (!geometry) continue;
        stage.scene.add(mask.flat
          ? new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
            vertexColors: true, transparent: true, opacity: 0.55,
          }))
          : new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({
            vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.12,
            depthWrite: false,
          })));
      }

      stage.frameShadow();
      stage.draw();
    };

    const draw = async ({ refit = true } = {}) => {
      const done = busy(holder, 'Reading geometry…');
      try {
        const summary = await readSummary();
        state.palette = state.palette ?? await entityPalette(ctx, objects);
        if (state.entities === null) {
          // Start with the labelled structures: the masks are big, and mostly in the way.
          state.entities = new Set(
            summary.entities.filter((e) => e.instances > 0).map((e) => e.name)
          );
          if (!state.entities.size) state.entities = new Set(state.palette.order);
        }
        if (state.maxGap === null) {
          state.maxGap = summary.widestGap;
          state.gap = summary.widestGap === null ? null : summary.widestGap * 0.25;
        }
        loaded = await readGeometry(summary);
        if (!_objectStage) {
          _objectStage = makeStage(await loadThree(), canvas);
          new ResizeObserver(() => {
            _objectStage?.resize(holder.clientWidth, holder.clientHeight);
          }).observe(holder);
        }
        paint();
        _objectStage.resize(holder.clientWidth, holder.clientHeight);
        if (refit) _objectStage.fit();

        const drawn = loaded.instances.length;
        const total = summary.entities
          .filter((e) => state.entities.has(e.name))
          .reduce((sum, e) => sum + e.instances, 0);
        footer.textContent = `${drawn.toLocaleString()} instance(s) drawn`
          + (drawn < total ? ` of ${total.toLocaleString()}, the largest by volume` : '')
          + `, ${loaded.masks.length} mask(s), in ${state.object.id}.`;
        buildControls(summary);
      } catch (error) {
        holder.style.display = 'none';
        sourceProblem(container, error, redraw);
      } finally {
        done();
      }
    };

    function buildControls(summary) {
      bar.replaceChildren();
      if (objects.length > 1) {
        bar.appendChild(selector('Object', objects.map((c) => c.id), state.object.id, (value) => {
          state.object = objects.find((c) => c.id === value);
          state.entities = null;
          state.maxGap = null;
          draw();
        }, { labels: shortOf }));
      }

      const colourings = ['entity', 'metric'];
      if (summary.widestGap !== null) colourings.push('group');
      bar.appendChild(selector('Colour by',
        colourings, state.colourBy,
        (value) => {
          state.colourBy = value;
          buildControls(summary);
          if (value === 'group') draw({ refit: false });
          else paint();
        }));

      if (state.colourBy === 'metric') {
        const metrics = metricsAvailable;
        bar.appendChild(selector('Metric', metrics, state.metric, (value) => {
          state.metric = value;
          draw({ refit: false });
        }, { labels: new Map(metrics.map((m) => [m, labelFor(m)])) }));
      }
      if (state.colourBy === 'group' && state.maxGap) {
        bar.appendChild(slider('Gap', {
          min: 0, max: state.maxGap, step: state.maxGap / 100, value: state.gap,
          format: (v) => `${v.toFixed(3)} µm`,
        }, null, (value) => {
          state.gap = value;
          draw({ refit: false });
        }));
      }

      // Exploding needs each instance's direction from the object centre, which rides in the
      // geometry file. Geometry written before that was carried has none, and a slider that
      // silently does nothing is worse than no slider.
      if (loaded?.instances.some((item) => item.polar.dist > 0)) {
        bar.appendChild(slider('Explode', {
          min: 0, max: 2, step: 0.1, value: state.explode,
          format: (v) => `${v.toFixed(1)}×`,
        }, null, (value) => {
          state.explode = value;
          paint();
          _objectStage?.fit();
        }));
      } else {
        const missing = document.createElement('span');
        missing.style.cssText = 'font-size:.78rem;color:#a15c00';
        missing.textContent = 'No explode: this geometry carries no polarity. '
          + 'Re-run pixel-patrol-anatomy mesh.';
        bar.appendChild(missing);
      }

      const chips = document.createElement('div');
      chips.style.cssText = 'display:flex;flex-wrap:wrap;gap:.5rem;align-items:center';
      for (const entity of summary.entities) {
        const swatch = document.createElement('span');
        swatch.style.cssText =
          `display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:.3rem;`
          + `background:${state.palette.of(entity.name)}`;
        const label = document.createElement('label');
        label.style.cssText = 'display:inline-flex;align-items:center;gap:.25rem;font-size:.78rem';
        label.title = [
          entity.instances ? `${entity.instances.toLocaleString()} instance(s)` : '',
          entity.masks ? 'whole-structure mask' : '',
        ].filter(Boolean).join(', ');
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = state.entities.has(entity.name);
        box.addEventListener('change', () => {
          if (box.checked) state.entities.add(entity.name);
          else state.entities.delete(entity.name);
          draw({ refit: false });
        });
        label.append(box, swatch, entity.name);
        chips.appendChild(label);
      }
      bar.appendChild(chips);
    }

    await draw();
  },
};

// ── Instance gallery ──────────────────────────────────────────────────────────

let _thumbStage = null;

/** One small renderer, reused for every thumbnail: a canvas each would exhaust WebGL. */
async function thumbStage() {
  if (!_thumbStage) {
    const canvas = document.createElement('canvas');
    canvas.width = THUMB_PX * 2;
    canvas.height = THUMB_PX * 2;
    _thumbStage = makeStage(await loadThree(), canvas, { shadows: false });
    _thumbStage.resize(THUMB_PX * 2, THUMB_PX * 2);
  }
  return _thumbStage;
}

/**
 * The material for an instance's own geometry.
 *
 * A mesh with a skeleton inside it is drawn see-through: an opaque surface hides the very
 * thing the skeleton metrics are about. Outlines need no such help.
 */
function instanceMaterial(THREE, { flat, skeleton }) {
  if (flat) return new THREE.LineBasicMaterial({ vertexColors: true });
  return new THREE.MeshPhongMaterial({
    vertexColors: true, side: THREE.DoubleSide, shininess: 35,
    ...(skeleton ? { transparent: true, opacity: 0.35, depthWrite: false } : {}),
  });
}


async function paintThumb(target, item, hex) {
  const stage = await thumbStage();
  const { THREE } = stage;
  stage.clear();
  const geometry = mergeInstances(THREE, [{ geometry: item.geometry, rgb: toRgb(hex) }]);
  if (!geometry) return;
  const material = instanceMaterial(THREE, item);
  stage.scene.add(item.flat
    ? new THREE.LineSegments(geometry, material)
    : new THREE.Mesh(geometry, material));
  if (item.skeleton) {
    const lines = mergeInstances(THREE, [{ geometry: item.skeleton, rgb: [0.1, 0.1, 0.12] }]);
    if (lines) {
      stage.scene.add(new THREE.LineSegments(lines,
        new THREE.LineBasicMaterial({ vertexColors: true })));
    }
  }
  stage.fit();
  const context = target.getContext('2d');
  context.clearRect(0, 0, target.width, target.height);
  context.drawImage(stage.renderer.domElement, 0, 0, target.width, target.height);
}

let _modalStage = null;

/** The full-size look at one instance, with its skeleton if it has one. */
async function openInstance(item, title) {
  const overlay = document.createElement('div');
  overlay.style.cssText =
    'position:fixed;inset:0;background:rgba(12,12,18,.72);z-index:9999;display:flex;'
    + 'align-items:center;justify-content:center';
  const panel = document.createElement('div');
  panel.style.cssText =
    'background:#f7f7fa;border-radius:8px;padding:.6rem;width:min(78vw,860px);'
    + 'height:min(78vh,640px);display:flex;flex-direction:column;gap:.4rem';
  const head = document.createElement('div');
  head.style.cssText = 'display:flex;align-items:center;gap:.6rem;color:#333;font-size:.8rem';
  head.append(title);
  const close = document.createElement('button');
  close.textContent = 'Close';
  close.style.cssText = 'margin-left:auto;font-size:.78rem;padding:.15rem .5rem';
  head.appendChild(close);
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'flex:1;width:100%;min-height:0;border-radius:6px';
  panel.append(head, canvas);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  const dismiss = () => {
    _modalStage?.clear();
    overlay.remove();
  };
  close.addEventListener('click', dismiss);
  overlay.addEventListener('click', (event) => { if (event.target === overlay) dismiss(); });

  const THREE = await loadThree();
  // The modal keeps its own stage across openings, for the same reason the gallery does.
  if (!_modalStage || _modalStage.renderer.domElement !== canvas) {
    _modalStage?.dispose();
    _modalStage = makeStage(THREE, canvas, { shadows: false });
  }
  const stage = _modalStage;
  stage.clear();
  const geometry = mergeInstances(THREE, [{ geometry: item.geometry, rgb: toRgb(item.hex) }]);
  if (geometry) {
    const material = instanceMaterial(THREE, item);
    stage.scene.add(item.flat
      ? new THREE.LineSegments(geometry, material)
      : new THREE.Mesh(geometry, material));
  }
  if (item.skeleton) {
    const lines = mergeInstances(THREE, [{ geometry: item.skeleton, rgb: [0.1, 0.1, 0.12] }]);
    if (lines) {
      stage.scene.add(new THREE.LineSegments(lines, new THREE.LineBasicMaterial({
        vertexColors: true,
      })));
    }
  }
  stage.resize(canvas.clientWidth, canvas.clientHeight);
  stage.fit();
}

/**
 * Read the biggest drawable rows of one kind for a preview.
 *
 * Bounded twice, by row count and by the vertices in the payload headers, because a tile
 * must stay cheap: the header columns exist so this can be decided without reading geometry.
 */
async function previewGeometry(ctx, objects, { rowType, limit, maxVertices, perEntity = 0 }) {
  // perEntity spreads the sample over the structures instead of taking the largest few, which
  // in a real object are all the same structure.
  const spread = perEntity
    ? `QUALIFY row_number() OVER (PARTITION BY "entity_name"
                                 ORDER BY ${GEOMETRY_SIZE} DESC NULLS LAST) <= ${perEntity}`
    : '';
  const rows = await ctx.queryRows(
    `SELECT "entity_name" AS entity, "mesh" AS mesh, "outline" AS outline
     FROM ${sourceOf(objects)}
     WHERE "row_type" = '${rowType}' AND ${HAS_GEOMETRY}
       AND COALESCE("mesh_vertices", "outline_vertices", 0) < ${maxVertices}
     ${spread}
     ORDER BY ${GEOMETRY_SIZE} DESC NULLS LAST
     LIMIT ${limit}`
  );
  return rows.flatMap((row) => {
    const drawable = decodeDrawable(row);
    return drawable ? [{ ...drawable, entity: String(row.entity) }] : [];
  });
}

/** Copy whatever the shared stage currently holds into a 2D canvas of its own. */
async function captureStage(stage, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(stage.renderer.domElement, 0, 0, width, height);
  return canvas;
}

/** One scene holding every item, in their structure colours: the object as it sits. */
async function paintScene(container, items, palette) {
  const stage = await thumbStage();
  const { THREE } = stage;
  stage.clear();
  for (const item of items) {
    const geometry = mergeInstances(THREE, [{
      geometry: item.geometry, rgb: toRgb(palette.of(item.entity)),
    }]);
    if (!geometry) continue;
    stage.scene.add(item.flat
      ? new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true }))
      : new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({
        vertexColors: true, side: THREE.DoubleSide, shininess: 30,
        transparent: true, opacity: 0.55, depthWrite: false,
      })));
  }
  stage.fit();
  const canvas = await captureStage(stage, THUMB_PX * 2, THUMB_PX * 2);
  canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain';
  container.appendChild(canvas);
  return true;
}

/** A few instances as their own cards: the gallery, in miniature. */
async function paintMiniGallery(container, items, palette) {
  const grid = document.createElement('div');
  const columns = items.length <= 4 ? 2 : 3;
  grid.style.cssText = 'position:absolute;inset:0;display:grid;gap:3px;'
    + `grid-template-columns:repeat(${columns}, 1fr)`;
  container.appendChild(grid);
  const stage = await thumbStage();
  const { THREE } = stage;
  for (const item of items) {
    stage.clear();
    const geometry = mergeInstances(THREE, [{
      geometry: item.geometry, rgb: toRgb(palette.of(item.entity)),
    }]);
    if (!geometry) continue;
    stage.scene.add(item.flat
      ? new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true }))
      : new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({
        vertexColors: true, side: THREE.DoubleSide, shininess: 35,
      })));
    stage.fit();
    const card = await captureStage(stage, THUMB_PX, THUMB_PX);
    card.style.cssText = 'width:100%;height:100%;object-fit:contain;border-radius:3px;'
      + 'background:#f7f7fa';
    grid.appendChild(card);
  }
  return grid.childElementCount > 0;
}

const instanceGallery = {
  id: 'anatomy-gallery',
  label: 'Instance Gallery',
  group: 'Visualization',
  info: [
    'The instances themselves. **Highest** and **lowest** put the tails of a metric on screen, '
    + 'where a sphericity outlier turns out to be two granules the segmentation merged. '
    + '**Random** samples the structure as it is, then shows the sample in metric order.',
    '',
    'Each instance is drawn in its structure\'s colour and framed in its group\'s; one with a '
    + 'skeleton is drawn see-through so the skeleton shows. The sample spans every object the '
    + 'filter leaves, or one group if you pick one. Click any of them to look properly.',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.includes('mesh_geometry_file');
  },

  // No message: the grid of instances is the message.
  async overviewPlot(container, ctx) {
    const objects = await geometryObjects(ctx);
    if (!objects.length) return false;
    const items = await previewGeometry(ctx, objects.slice(0, 1),
                                       { rowType: 'instance', limit: 6, maxVertices: 20_000 });
    if (!items.length) return false;
    return paintMiniGallery(container, items, await entityPalette(ctx, objects));
  },

  async render(container, ctx) {
    const objects = await geometryObjects(ctx);
    if (!objects.length) {
      emptyState(container,
        'No geometry beside this report. Re-run with --with-mesh (or `anatomy mesh`) to '
        + 'write one geometry.parquet per object.');
      return;
    }
    const objectById = new Map(objects.map((c) => [c.id, c]));

    const bar = controlBar(container);
    const grid = document.createElement('div');
    grid.style.cssText =
      `display:grid;grid-template-columns:repeat(auto-fill,minmax(${THUMB_PX}px,1fr));gap:10px`;
    container.appendChild(grid);
    const footer = note(container, '');

    const state = {
      entity: null, metric: metricsFor(ctx.schema)[0], order: 'highest',
      rows: GALLERY_ROWS[0], palette: null, group: ALL_GROUPS,
    };

    /**
     * How many thumbnails to ask for: whole rows of however many the card fits.
     *
     * The grid is auto-fill, so its column count is a property of the width, not of this widget.
     * Asking for a round number instead left the last row part empty at every width that did not
     * happen to divide it.
     */
    const wanted = () => {
      const columns = Math.max(
        1, getComputedStyle(grid).gridTemplateColumns.split(' ').filter(Boolean).length);
      return Math.min(MAX_THUMBS - (MAX_THUMBS % columns), columns * state.rows);
    };

    // The objects the group filter leaves. Reading one group's geometry is also the cheap
    // way to look at a condition on its own, since only those files are opened.
    const inScope = () => (state.group === ALL_GROUPS
      ? objects
      : objects.filter((o) => String(o.group) === String(state.group)));

    const redraw = () => {
      container.replaceChildren();
      instanceGallery.render(container, ctx);
    };

    const draw = async () => {
      grid.replaceChildren();
      const loading = document.createElement('p');
      loading.style.cssText = 'font-size:.8rem;color:#666';
      loading.textContent = 'Reading geometry…';
      grid.appendChild(loading);
      try {
        const scoped = inScope();
        if (!scoped.length) {
          grid.replaceChildren();
          emptyState(grid, `No object in ${ctx.groupLabel(state.group)} has geometry.`);
          return;
        }
        const source = sourceOf(scoped);
        if (state.entity === null) {
          const entities = await ctx.queryRows(
            `SELECT DISTINCT "entity_name" AS name FROM ${source}
             WHERE "row_type" = 'instance' AND ${HAS_GEOMETRY} ORDER BY 1`
          );
          if (!entities.length) {
            grid.replaceChildren();
            emptyState(grid, 'No instance meshes in the geometry for these objects.');
            return;
          }
          state.entity = focusedStructure(entities.map((r) => String(r.name)));
        }

        const metric = ctx.sql.q(state.metric);
        const [totals] = await ctx.queryRows(
          `SELECT COUNT(*) AS n FROM ${source}
           WHERE "row_type" = 'instance' AND ${HAS_GEOMETRY}
             AND "entity_name" = ${esc(state.entity)} AND isfinite(${metric})`
        );
        // "random" picks the sample first and sorts it afterwards, so the grid reads in
        // metric order either way: the tails are a different question from a fair sample.
        const pick = state.order === 'random'
          ? `ORDER BY random() LIMIT ${wanted()}`
          : `ORDER BY ${metric} ${state.order === 'lowest' ? 'ASC' : 'DESC'} `
            + `LIMIT ${wanted()}`;
        const rows = await ctx.queryRows(
          `SELECT * FROM (
             SELECT "object_id" AS object_id, "label_id" AS label, ${metric} AS value,
                    "mesh" AS mesh, "outline" AS outline, "skeleton" AS skeleton
             FROM ${source}
             WHERE "row_type" = 'instance' AND ${HAS_GEOMETRY}
               AND "entity_name" = ${esc(state.entity)} AND isfinite(${metric})
             ${pick}
           ) ORDER BY value DESC`
        );

        grid.replaceChildren();
        if (!rows.length) {
          emptyState(grid, `No ${state.entity} instance has a ${labelFor(state.metric)}.`);
          return;
        }
        // Two encodings: the shape takes the structure's colour, the border says which
        // group the object is in.
        state.palette = state.palette ?? await entityPalette(ctx, objects);
        const meshHex = state.palette.of(state.entity);
        for (const row of rows) {
          const drawable = decodeDrawable(row);
          if (!drawable) continue;
          const { geometry, flat } = drawable;
          const object = objectById.get(String(row.object_id));
          const groupHex = ctx.color.group(object?.group);
          const card = document.createElement('figure');
          card.style.cssText =
            `margin:0;border:2px solid ${groupHex};border-radius:6px;overflow:hidden;`
            + 'cursor:pointer;background:#f7f7fa';
          const thumb = document.createElement('canvas');
          thumb.width = THUMB_PX * 2;
          thumb.height = THUMB_PX * 2;
          thumb.style.cssText = 'width:100%;aspect-ratio:1;display:block';
          const caption = document.createElement('figcaption');
          caption.style.cssText = 'font-size:.7rem;padding:.25rem .35rem;line-height:1.3;color:#333';
          const value = num(row.value);
          caption.textContent = `${state.entity} #${row.label} · `
            + `${Number.isFinite(value) ? value.toPrecision(3) : '-'}`;
          caption.title = `${object?.id ?? ''}: ${labelFor(state.metric)} ${value}`;
          card.append(thumb, caption);
          const skeleton = decodePayload(row.skeleton, 2);
          card.addEventListener('click', () => openInstance(
            { geometry, flat, skeleton, hex: meshHex },
            `${object?.id ?? ''} · ${state.entity} #${row.label} · `
            + `${labelFor(state.metric)} ${Number.isFinite(value) ? value.toPrecision(4) : '-'}`));
          grid.appendChild(card);
          // Sequential: one WebGL stage draws them all, and each needs the canvas to itself.
          await paintThumb(thumb, { geometry, flat, skeleton }, meshHex);
        }
        const scope = state.group === ALL_GROUPS
          ? `${scoped.length} object(s)` : `${ctx.groupLabel(state.group)}`;
        footer.textContent =
          `${rows.length} of ${Number(totals?.n ?? 0).toLocaleString()} ${state.entity} `
          + `instances, ${state.order === 'random' ? 'a random sample' : state.order} `
          + `by ${labelFor(state.metric).toLowerCase()}, across ${scope}.`;
      } catch (error) {
        grid.replaceChildren();
        sourceProblem(container, error, redraw);
      }
    };

    const buildControls = async () => {
      const source = sourceOf(objects);
      let entities = [];
      try {
        entities = (await ctx.queryRows(
          `SELECT DISTINCT "entity_name" AS name FROM ${source}
           WHERE "row_type" = 'instance' AND ${HAS_GEOMETRY} ORDER BY 1`
        )).map((r) => String(r.name));
      } catch { /* draw() reports the problem and offers the fix */ }
      bar.replaceChildren();
      if (ctx.groups.length > 1) {
        const groups = [ALL_GROUPS, ...ctx.groups.map(String)];
        const labels = new Map([[ALL_GROUPS, 'all'],
                               ...ctx.groups.map((g) => [String(g), ctx.groupLabel(g)])]);
        bar.appendChild(selector('Group', groups, String(state.group), (value) => {
          state.group = value;
          draw();
        }, { labels }));
      }
      if (entities.length) {
        state.entity = state.entity ?? focusedStructure(entities);
        bar.appendChild(selector('Structure', entities, state.entity, (value) => {
          state.entity = value;
          focusStructure(value);
          draw();
        }));
      }
      const metrics = metricsFor(ctx.schema);
      bar.appendChild(selector('Sort by', metrics, state.metric, (value) => {
        state.metric = value;
        draw();
      }, { labels: new Map(metrics.map((m) => [m, labelFor(m)])) }));
      bar.appendChild(selector('Pick', ['highest', 'lowest', 'random'], state.order, (value) => {
        state.order = value;
        draw();
      }));
      bar.appendChild(selector('Rows', GALLERY_ROWS, state.rows, (value) => {
        state.rows = Number(value);
        draw();
      }));
      // The card borders are group colours, which needs saying.
      ctx.plot.renderDomGroupLegend(bar, { minGroups: 2 });
    };

    onStructureFocus(container, async (name) => {
      if (name === state.entity) return;
      state.entity = name;
      await buildControls();
      await draw();
    });
    await buildControls();
    await draw();
  },
};

export default [object3d, instanceGallery];

// Exported for the test suite: a binary contract with mesh.py (decodePayload), index
// arithmetic (mergeInstances), and what the object view colours as touching (contactGroups).
export { decodePayload, mergeInstances, contactGroups, explodeOffset };
