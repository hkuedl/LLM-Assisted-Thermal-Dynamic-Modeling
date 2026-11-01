"""Public OOD post-processing configuration.

The release layout is expected to contain sibling ``Data``, ``Result`` and
``public code`` directories.  Set ``BUILDING_RESULT_ROOT`` only when using a
different layout.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import re

RELEASE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get('BUILDING_DATA_ROOT', RELEASE_ROOT / 'Data'))
FINAL_ROOT = Path(os.environ.get('BUILDING_RESULT_ROOT', RELEASE_ROOT / 'Result'))
# Public outputs remain inside Result and do not overwrite the input manifests.
RUN_ROOT = None  # None -> FINAL_ROOT/OOD/postprocessing_public
SAMPLES = 10
PROMPT_STYLE = 'public'  # original reproduces the historical wording
BUILDINGS = []  # [] = all buildings present in the existing index
OODS = ['OOD-2', 'OOD-3', 'OOD-4', 'OOD-5']
TRAIN_DAYS = ['14-day', '21-day', '28-day', 'Full']
MAX_WORKERS = 6
ZONES_PER_CALL = 50
MAX_ATTEMPTS = 3  # total attempts per batch per execution; not additional samples
PROVIDER = 'together'
MODEL = os.environ.get('LLM_MODEL', 'meta-models/Muse-Glimmer-30B')
TEMPERATURE = 0.2
# ``generated`` reads/writes the public mean-N priors under RUN_ROOT.
PRIOR_SOURCE = 'generated'  # existing | generated
EXISTING_PRIORS = None  # optional frozen reference priors
# These indexes are preserved when the complete Final Results folder is renamed Result.
CASE_INDEX = None  # None -> FINAL_ROOT/OOD/postprocessing/case_manifest.csv
INPUT_INDEX = None  # None -> FINAL_ROOT/OOD/postprocessing/run_manifest.csv
FIXED_STRENGTHS = [i / 20 for i in range(21)]
# Historical results used the original column matching/alignment. Keep explicit.
TARGET_MATCHING = 'legacy'  # legacy | exact_air (a separately labelled evaluation)
SAVE_CORRECTED = False
SAVE_ZONE_METRICS = False  # optional large per-zone tables; does not change case scores
CHECKPOINT_EVERY = 50  # offline cases between progress snapshots, not API batch size

KEYS = ['Architecture', 'Method', 'Model_Tag', 'Seed', 'Building',
        'Train_Days', 'OOD_Label', 'External_Variant']
OURS = 'LLM event+phase'


def settings(**overrides):
    s = {name.lower(): globals()[name] for name in (
        'FINAL_ROOT', 'RUN_ROOT', 'SAMPLES', 'PROMPT_STYLE', 'BUILDINGS', 'OODS', 'TRAIN_DAYS',
        'MAX_WORKERS', 'ZONES_PER_CALL', 'MAX_ATTEMPTS', 'PROVIDER', 'MODEL',
        'TEMPERATURE', 'PRIOR_SOURCE', 'EXISTING_PRIORS', 'CASE_INDEX',
        'INPUT_INDEX', 'FIXED_STRENGTHS', 'TARGET_MATCHING', 'SAVE_CORRECTED',
        'SAVE_ZONE_METRICS', 'CHECKPOINT_EVERY')}
    s.update(overrides)
    s['final_root'] = Path(s['final_root']).expanduser()
    root = s['final_root']
    defaults = {
        'run_root': root / 'OOD/postprocessing_public',
        'case_index': root / 'OOD/postprocessing/case_manifest.csv',
        'input_index': root / 'OOD/postprocessing/run_manifest.csv',
        'existing_priors': root / 'OOD/postprocessing_reference/priors',
    }
    for key, value in defaults.items():
        s[key] = Path(s[key]).expanduser() if s.get(key) is not None else value
    for key in ('samples', 'max_workers', 'zones_per_call', 'max_attempts', 'checkpoint_every'):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if s['prompt_style'] not in ('original', 'public'):
        raise ValueError('prompt_style must be original or public')
    if s['prior_source'] not in ('existing', 'generated'):
        raise ValueError('prior_source must be existing or generated')
    if s['target_matching'] not in ('legacy', 'exact_air'):
        raise ValueError('target_matching must be legacy or exact_air')
    if s['provider'] not in ('together', 'openai'):
        raise ValueError('provider must be together or openai')
    for value in s['fixed_strengths']:
        if not 0 <= float(value) <= 1:
            raise ValueError('fixed_strengths must be in [0,1]')
    # This is a dedicated new run, never a reset of saved public results.
    if s['run_root'].resolve() in (root.resolve(), (root/'OOD').resolve(),
                                   (root/'OOD/postprocessing').resolve()):
        raise ValueError('run_root must be a separate output subdirectory')
    return s


def cli():
    p = argparse.ArgumentParser(description='Use config.py defaults; override only what you need.')
    for flag in ('final-root', 'run-root', 'existing-priors', 'case-index', 'input-index'):
        p.add_argument('--'+flag, type=Path)
    for flag in ('samples', 'max-workers', 'zones-per-call', 'max-attempts'):
        p.add_argument('--'+flag, type=int)
    for flag in ('buildings', 'oods', 'train-days'):
        p.add_argument('--'+flag, help='comma-separated values')
    for flag in ('provider', 'model', 'prior-source', 'target-matching', 'prompt-style'):
        p.add_argument('--'+flag)
    p.add_argument('--temperature', type=float)
    p.add_argument('--run', action='store_true', help='Actually call LLM (generation only)')
    p.add_argument('--force', action='store_true', help='Recompute offline metrics, never LLM caches')
    args = vars(p.parse_args())
    for key in ('buildings', 'oods', 'train_days'):
        if args.get(key) is not None:
            args[key] = [x.strip() for x in args[key].split(',') if x.strip()]
    run, force = args.pop('run'), args.pop('force')
    return settings(**{k:v for k,v in args.items() if v is not None}), run, force


def read_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        # Name the file without logging its contents or any stored credentials.
        raise json.JSONDecodeError(str(path) + ': ' + exc.msg,
                                   exc.doc, exc.pos) from None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def read_index(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Existing index not found: '+str(path)+
            '\nSet case_index/input_index to its actual location; no search or migration is run.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def resolve_input(value, index, root):
    if not str(value).strip():
        raise ValueError('Empty input path in '+str(index))
    path = Path(str(value)).expanduser()
    options = [path] if path.is_absolute() else [root/path, Path(index).parent/path]
    # Manifests saved on the experiment server may contain absolute paths.  A
    # released copy keeps the same suffix below Result/ or Data/.
    if path.is_absolute():
        parts = path.parts
        if 'Building_Temperature_Final_Results' in parts:
            position = parts.index('Building_Temperature_Final_Results')
            options.append(root.joinpath(*parts[position+1:]))
        if 'Data' in parts:
            position = parts.index('Data')
            options.append(DATA_ROOT.joinpath(*parts[position+1:]))
    for option in dict.fromkeys(options):
        if option.is_file():
            return option.resolve()
    raise FileNotFoundError('Input not found for %r (manifest: %s)' % (value, index))


def selected(row, s):
    building = row.get('Building', row.get('building'))
    ood = row.get('OOD_Label', row.get('ood'))
    day = row.get('Train_Days')
    return ((not s['buildings'] or building in s['buildings']) and ood in s['oods']
            and (day is None or day in s['train_days']))


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._') or 'unnamed'


def prior_path(s, building, ood):
    if s['prior_source'] == 'existing':
        # Exact layout of the selected pre-packaging experiment, not public/current.
        return s['existing_priors'] / building / 'Full' / ood / 'shared_prior.json'
    return generation_root(s) / 'priors' / ('samples_%d'%s['samples']) / building / ood / 'shared_prior.json'


def metric_root(s):
    source = 'existing' if s['prior_source']=='existing' else 'samples_%d'%s['samples']
    root = s['run_root'] if s['prior_source']=='existing' else generation_root(s)
    return root / 'metrics' / source / s['target_matching']


def generation_root(s):
    # Keep the historical and public prompt outputs separate.
    return s['run_root'] if s['prompt_style']=='original' else s['run_root']/'public_prompt'
