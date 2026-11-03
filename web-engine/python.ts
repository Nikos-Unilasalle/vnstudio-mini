/**
 * Real Python for the Python node, through Pyodide (CPython compiled to WASM).
 *
 * The desktop node `exec`s its script with numpy, pandas and OpenCV at hand, and
 * the course material leans on that — the ML exercises are pandas from end to
 * end. Pyodide is pulled from the CDN the first time a Python script runs, never
 * before: it is ~15 MB with pandas, which no graph without a Python node should
 * pay for. The browser caches it afterwards.
 *
 * Pyodide's version tracks the desktop's pandas major (3.x): pandas 2 → 3 changed
 * copy and string semantics enough that an exercise could behave differently.
 *
 * The script runs under the same contract as engine/plugins/logic_python.py:
 * inputs as `a`, `b`, `c`…, `np`/`pd`/`cv2`/`state` in scope, the same blocked
 * imports and builtins, every `out_*` variable becoming an output, and the error
 * on `__error__` for the editor.
 */
import { dfMeta, isDf, makeDf } from './dataframe'
import { isMat } from './cvUtils'
import type { RunContext } from './types'

const PYODIDE_VERSION = '314.0.7'
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

/** Mirrors the desktop node's sandbox, then adds the converters both ways. */
const PRELUDE = String.raw`
import builtins as _builtins, traceback as _traceback, os as _os, sys as _sys
import numpy as np
import pandas as pd

# matplotlib, if a script imports it, must draw off-screen: there is no window
# in a worker. Figures reach the canvas through _vn_figure below.
_os.environ['MPLBACKEND'] = 'Agg'

_BLOCKED_IMPORTS = frozenset([
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'http', 'urllib',
    'requests', 'pathlib', 'glob', 'importlib', 'ctypes', 'mmap',
    'builtins', 'io', 'pty', 'atexit', 'signal', 'threading', 'multiprocessing',
])

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = (name or '').split('.')[0]
    if top in _BLOCKED_IMPORTS:
        raise ImportError(f"Le module '{top}' est bloqué dans la node Python : il donne accès au système ou au réseau.")
    return _builtins.__import__(name, globals, locals, fromlist, level)

_ALLOWED = {
    'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'chr', 'complex',
    'dict', 'divmod', 'enumerate', 'filter', 'float', 'format',
    'frozenset', 'getattr', 'hasattr', 'hash', 'hex', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next',
    'oct', 'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round',
    'set', 'setattr', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
    'ArithmeticError', 'AttributeError', 'Exception', 'IndexError', 'KeyError',
    'NotImplementedError', 'OverflowError', 'RuntimeError', 'StopIteration',
    'TypeError', 'ValueError', 'ZeroDivisionError',
}
_SAFE_BUILTINS = {k: getattr(_builtins, k) for k in _ALLOWED}
_SAFE_BUILTINS['__import__'] = _safe_import

_STATES = {}

def _vn_df(columns, rows):
    return pd.DataFrame(rows.to_py(), columns=list(columns.to_py()))

def _vn_array(buffer, shape, dtype):
    return np.frombuffer(buffer.to_bytes(), dtype=dtype).reshape(tuple(shape.to_py())).copy()

def _vn_plain(v):
    """Python value → something toJs can carry: natives, lists, dicts."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, (pd.Timestamp, pd.Timedelta)):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _vn_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_vn_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_vn_plain(x) for x in v.tolist()]
    return str(v)

def _vn_frame(df):
    # The web DataFrame has no index: a meaningful one (a groupby key, column
    # names after df.mean()) becomes a column instead of vanishing. Bare row
    # numbers left by a filter carry nothing, and as a column they would turn
    # into a bogus variable downstream, so those are dropped.
    idx = df.index
    if not isinstance(idx, pd.RangeIndex) or idx.name is not None:
        row_numbers = idx.name is None and not isinstance(idx, pd.MultiIndex) and pd.api.types.is_integer_dtype(idx.dtype)
        df = df.reset_index(drop=row_numbers)
    columns = [str(c) for c in df.columns]
    rows = [[_vn_plain(x) for x in row] for row in df.itertuples(index=False, name=None)]
    return {'kind': 'df', 'columns': columns, 'rows': rows}

def _vn_figure(v):
    """A matplotlib figure (or pyplot, or an Axes, or a grid of Axes) → BGR image, else None."""
    if 'matplotlib' not in _sys.modules:
        return None
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    plt = _sys.modules.get('matplotlib.pyplot')
    if plt is not None and v is plt:
        v = plt.gcf()
    if isinstance(v, np.ndarray) and v.dtype == object and v.size and isinstance(v.flat[0], Axes):
        v = v.flat[0]
    if isinstance(v, Axes):
        v = v.figure
    if not isinstance(v, Figure):
        return None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(v)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, 2::-1])  # RGBA → BGR, the engine's convention

def _vn_is_figure(v):
    if 'matplotlib' not in _sys.modules:
        return False
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    if isinstance(v, np.ndarray) and v.dtype == object and v.size:
        v = v.flat[0]
    return v is _sys.modules.get('matplotlib.pyplot') or isinstance(v, (Figure, Axes))

def _vn_export(v):
    image = _vn_figure(v)
    if image is not None:
        v = image
    if isinstance(v, pd.DataFrame):
        return _vn_frame(v)
    if isinstance(v, pd.Series):
        return _vn_frame(v.to_frame(name=v.name if v.name is not None else 'value'))
    if isinstance(v, np.ndarray) and v.ndim in (2, 3) and v.size > 0 and (v.ndim == 2 or v.shape[2] <= 4):
        a = v
        if a.dtype == np.bool_:
            a = a.astype(np.uint8) * 255
        elif a.dtype not in (np.uint8, np.int32, np.float32, np.float64):
            a = a.astype(np.float32 if np.issubdtype(a.dtype, np.floating) else np.int32)
        a = np.ascontiguousarray(a)
        return {'kind': 'mat', 'dtype': str(a.dtype), 'shape': list(a.shape), 'data': a.tobytes()}
    return {'kind': 'value', 'value': _vn_plain(v)}

_NOT_HERE = {
    'open': "lire ou écrire un fichier est impossible ici : les données arrivent par les entrées a, b, c…",
    'input': "un script ne peut pas poser de question au clavier : branchez une node Number sur une entrée",
    'dir': "pour découvrir ce que contient a, renvoyez par exemple out_info = str(type(a)) ou out_colonnes = list(a.columns)",
    'help': "la documentation se lit dans le navigateur (pandas.pydata.org, scikit-learn.org)",
    'exit': "un script s'arrête tout seul à la dernière ligne",
    'quit': "un script s'arrête tout seul à la dernière ligne",
}
_FORBIDDEN = {'eval', 'exec', 'compile', 'globals', 'locals', 'vars', '__import__', 'breakpoint', 'memoryview', 'object', 'super', 'property', 'classmethod', 'staticmethod', 'callable', 'id'}

def _vn_hint(e, ctx):
    """A short French explanation of the error a beginner most likely made, or ''."""
    if isinstance(e, IndentationError):
        return "Indentation incorrecte : après une ligne qui finit par « : » (if, for, def…), les lignes du bloc doivent être décalées de 4 espaces, toutes pareil."
    if isinstance(e, SyntaxError):
        return "Erreur de syntaxe : vérifiez les parenthèses et guillemets fermés, et le « : » en fin de if / for / def."
    if isinstance(e, NameError):
        name = getattr(e, 'name', None)
        if name == '__build_class__' or '__build_class__' in str(e):
            return "Les classes (mot-clé class) ne sont pas disponibles dans la node Python : utilisez des fonctions (def)."
        if name in _NOT_HERE:
            return f"{name}() n'est pas disponible dans la node Python : {_NOT_HERE[name]}."
        if name in _FORBIDDEN:
            return f"{name} est bloqué dans la node Python pour des raisons de sécurité."
        if name and len(name) == 1 and name.isalpha() and name.islower():
            return f"Rien n'est branché sur l'entrée « {name} » : reliez une node au « + » d'entrée (la 1re devient a, la 2e b, etc.)."
        if name in ('plt', 'sns', 'math', 'random', 'sklearn', 'scipy'):
            return f"« {name} » n'est pas importé : ajoutez l'import en haut du script (ex. import matplotlib.pyplot as plt)."
        if name:
            return f"La variable « {name} » n'existe pas : faute de frappe ? majuscule ? variable créée plus bas dans le script ?"
    if isinstance(e, KeyError) and e.args and isinstance(e.args[0], str):
        key = e.args[0]
        for var, value in ctx.items():
            if isinstance(value, pd.DataFrame) and not var.startswith('_') and key not in value.columns:
                cols = ', '.join(repr(str(c)) for c in list(value.columns)[:12])
                more = '…' if len(value.columns) > 12 else ''
                return f"La colonne « {key} » n'existe pas. Colonnes de {var} : {cols}{more}. Recopiez le nom exactement (majuscules, espaces, unités)."
    if isinstance(e, ModuleNotFoundError):
        return f"Le module « {e.name} » n'existe pas dans le navigateur : vérifiez l'orthographe (numpy, pandas, scikit-learn → sklearn, scipy et matplotlib sont disponibles)."
    if isinstance(e, AttributeError) and "'DataFrame' object has no attribute" in str(e):
        return "Le tableau n'a ni méthode ni colonne de ce nom : vérifiez l'orthographe. Pour une colonne, écrivez a['nom de la colonne']."
    if isinstance(e, ValueError) and 'could not convert string to float' in str(e):
        return "Un calcul a reçu du texte au lieu de nombres : retirez les colonnes de texte (ex. 'target') avant d'entraîner ou de calculer."
    if isinstance(e, ValueError) and 'Input X contains NaN' in str(e):
        return "Le modèle refuse les valeurs manquantes (NaN) : nettoyez d'abord, avec dropna() ou fillna()."
    return ''

def _vn_run(code, node_id, inputs):
    ctx = {'__builtins__': _SAFE_BUILTINS, 'np': np, 'pd': pd, 'state': _STATES.setdefault(node_id, {})}
    try:
        import cv2
        ctx['cv2'] = cv2
    except ImportError:
        pass
    # print() has no terminal to write to here: its text is collected instead
    # and comes back as the out_print console (see logic.ts).
    printed = []
    def _print(*args, sep=' ', end='\n', **_ignored):
        printed.append(sep.join(str(a) for a in args) + end)
    ctx['print'] = _print
    ctx.update(inputs)
    error = ''
    try:
        exec(compile(code, '<script>', 'exec'), ctx)
    except BaseException as e:
        line = None
        if isinstance(e, SyntaxError):
            line = e.lineno
        else:
            for frame in _traceback.extract_tb(e.__traceback__):
                if frame.filename == '<script>':
                    line = frame.lineno
        where = f"Ligne {line} — " if line else ''
        technical = f"{type(e).__name__}: {e}"
        hint = _vn_hint(e, ctx)
        error = where + (f"{hint} ({technical})" if hint else technical)
    outputs = {k: _vn_export(v) for k, v in ctx.items() if k.startswith('out_')}
    # A figure drawn with plt but never handed to an out_ variable is what a
    # notebook would show under plt.show(): it becomes out_plot.
    plt = _sys.modules.get('matplotlib.pyplot')
    if plt is not None:
        if plt.get_fignums() and 'out_plot' not in outputs and not any(_vn_is_figure(ctx[k]) for k in outputs):
            outputs['out_plot'] = _vn_export(plt.gcf())
        plt.close('all')
    used_print = 'print' in code
    return {'outputs': outputs, 'error': error, 'printed': ''.join(printed) if used_print or printed else None}
`

let loading: Promise<any> | null = null
let attempts = 0

/**
 * Runs a Pyodide loading step with `importScripts` behaving as it natively does
 * in a module worker — throwing. Pyodide probes it to refuse classic workers,
 * and the polyfill MediaPipe needs (importScriptsPolyfill.ts) makes the probe
 * succeed, so Pyodide would think it is in one.
 */
async function asModuleWorker<T>(step: () => Promise<T>): Promise<T> {
  const scope = self as any
  const polyfill = scope.importScripts
  scope.importScripts = () => {
    throw new TypeError('Module scripts don’t support importScripts().')
  }
  try {
    return await step()
  } finally {
    scope.importScripts = polyfill
  }
}
let cv2Loaded = false
let pyodideReady = false
let helpers: { array: any; df: any; run: any } | null = null

function loadPyodideOnce(ctx: RunContext): Promise<any> {
  if (!loading) {
    loading = (async () => {
      ctx.report(null, 'Chargement de Python (première utilisation, ~15 Mo)…')
      // A failed dynamic import stays failed for this worker's lifetime (the
      // module map caches it), so a retry has to ask for a distinct URL.
      const retry = attempts++ > 0 ? `?retry=${attempts}` : ''
      const module = await asModuleWorker(() => import(/* @vite-ignore */ `${PYODIDE_BASE}pyodide.mjs${retry}`))
      const pyodide: any = await asModuleWorker(() => module.loadPyodide({ indexURL: PYODIDE_BASE }))
      ctx.report(null, 'Chargement de numpy et pandas…')
      await asModuleWorker(() => pyodide.loadPackage(['numpy', 'pandas']))
      pyodide.runPython(PRELUDE)
      // Looked up once: every globals.get() hands back a fresh proxy to free.
      helpers = {
        array: pyodide.globals.get('_vn_array'),
        df: pyodide.globals.get('_vn_df'),
        run: pyodide.globals.get('_vn_run'),
      }
      return pyodide
    })()
    // A failed download must not poison every later run: let the next one retry.
    loading.catch(() => { loading = null })
  }
  return loading
}

const MAT_DTYPES: Record<number, { dtype: string; view: string }> = {}

function matToPython(cv: any, py: any, mat: any): any {
  if (!Object.keys(MAT_DTYPES).length) {
    MAT_DTYPES[cv.CV_8U] = { dtype: 'uint8', view: 'data' }
    MAT_DTYPES[cv.CV_32S] = { dtype: 'int32', view: 'data32S' }
    MAT_DTYPES[cv.CV_32F] = { dtype: 'float32', view: 'data32F' }
    MAT_DTYPES[cv.CV_64F] = { dtype: 'float64', view: 'data64F' }
  }
  let source = mat
  let temp: any = null
  if (!MAT_DTYPES[mat.depth()]) {
    temp = new cv.Mat()
    mat.convertTo(temp, cv.CV_32F)
    source = temp
  }
  const { dtype, view } = MAT_DTYPES[source.depth()]
  const channels = source.channels()
  const shape = channels > 1 ? [source.rows, source.cols, channels] : [source.rows, source.cols]
  // Copy out of the WASM heap: the view dies with the Mat.
  const buffer = (source[view] as ArrayLike<number> & { slice(): any }).slice()
  temp?.delete()
  return helpers!.array(buffer, shape, dtype)
}

function toPython(cv: any, py: any, value: unknown): any {
  if (isMat(value)) return matToPython(cv, py, value)
  if (isDf(value)) {
    // Row arrays, not objects: they cross into Python as plain lists.
    const rows = value.rows.map((r) => value.columns.map((c) => r[c] ?? null))
    return helpers!.df(value.columns, rows)
  }
  return py.toPy(value)
}

const CV_TYPES: Record<string, [string, string]> = {
  uint8: ['CV_8U', 'Uint8Array'],
  int32: ['CV_32S', 'Int32Array'],
  float32: ['CV_32F', 'Float32Array'],
  float64: ['CV_64F', 'Float64Array'],
}

function fromPython(ctx: RunContext, exported: any): unknown {
  if (exported.kind === 'df') {
    const columns: string[] = exported.columns
    const rows = (exported.rows as unknown[][]).map((row) => Object.fromEntries(columns.map((c, i) => [c, row[i]])))
    return makeDf(columns, rows)
  }
  if (exported.kind === 'mat') {
    const cv = ctx.cv
    const [depth, arrayType] = CV_TYPES[exported.dtype]
    const [rows, cols, channels = 1] = exported.shape as number[]
    const bytes: Uint8Array = exported.data
    // A fresh copy starts at offset 0, so the wider views are always aligned.
    const typed = new (globalThis as any)[arrayType](bytes.slice().buffer)
    const mat = ctx.track(new cv.Mat(rows, cols, cv[`${depth}C${channels}`]))
    mat[{ CV_8U: 'data', CV_32S: 'data32S', CV_32F: 'data32F', CV_64F: 'data64F' }[depth]!].set(typed)
    return mat
  }
  return exported.value
}

/** Runs a Python script for the node `ctx.nodeId`; returns its out_* values plus `__error__` (and `df_meta`, as on desktop). */
export async function runPython(code: string, inputs: Record<string, unknown>, ctx: RunContext): Promise<Record<string, unknown>> {
  const firstRun = !pyodideReady
  let py: any
  let reported = firstRun
  try {
    py = await loadPyodideOnce(ctx)
    // Whatever the script imports (scipy, sklearn…) is fetched before it runs.
    await asModuleWorker(() => py.loadPackagesFromImports(code))
    if (!cv2Loaded && /\bcv2\b/.test(code)) {
      ctx.report(null, 'Chargement d’OpenCV pour Python…')
      reported = true
      await asModuleWorker(() => py.loadPackage('opencv-python'))
      cv2Loaded = true
    }
  } catch (error) {
    ctx.report(null, '')
    const reason = error instanceof Error ? error.message : String(error)
    return { __error__: `Python n’a pas pu être chargé (connexion ?) — nouvel essai à la prochaine exécution. ${reason}` }
  }
  pyodideReady = true
  if (reported) ctx.report(null, '')

  const pyInputs = py.toPy({})
  for (const [name, value] of Object.entries(inputs)) {
    // As on desktop, every connected input that is an identifier becomes a variable.
    if (!/^[A-Za-z_]\w*$/.test(name) || name === 'raw_frame' || value === undefined) continue
    const converted = toPython(ctx.cv, py, value)
    pyInputs.set(name, converted)
    converted?.destroy?.()
  }

  const result = helpers!.run(code, ctx.nodeId, pyInputs)
  pyInputs.destroy()
  const plain = result.toJs({ dict_converter: Object.fromEntries })
  result.destroy()

  const outputs: Record<string, unknown> = { __error__: plain.error }
  if (typeof plain.printed === 'string') outputs.__printed__ = plain.printed
  for (const [name, exported] of Object.entries(plain.outputs as Record<string, any>)) {
    outputs[name] = fromPython(ctx, exported)
    if (outputs.df_meta === undefined && isDf(outputs[name])) outputs.df_meta = dfMeta(outputs[name] as any)
  }
  return outputs
}
