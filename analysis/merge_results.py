"""Merge the result folders of the separate Colab sessions into ONE canonical results folder.
Naming: 'R18-C' = Stage 6 execution of seed 42 (used in the 3-seed comparison);
        'R18-C-orig' = the earlier Stage 4 execution of the same configuration and seed (kept, not discarded)."""
import sys, shutil, re
from pathlib import Path
import pandas as pd
sp, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists(): shutil.rmtree(out)
for d in ['metrics', 'predictions']: (out / d).mkdir(parents=True)
SRC = {  # run id -> (folder, id inside that folder)
    **{e: ('baseline', e) for e in ['B1', 'B2']},
    **{e: ('r18orig', e) for e in ['R18-A', 'R18-B']}, 'R18-C-orig': ('r18orig', 'R18-C'),
    **{e: ('s6a', e) for e in ['R18-C', 'R18-C-s43', 'R18-C-s44', 'T1-lr1e-5', 'T2-lr1e-3', 'T3-bs16', 'T4-adamw', 'T5-dropout', 'T6-weighted', 'T7-hflip']},
    **{e: ('r50orig', e) for e in ['R50-A', 'R50-B', 'R50-C']}, **{e: ('s6b', e) for e in ['F-R50-s43', 'F-R50-s44']}}
n = 0
for new, (folder, old) in SRC.items():
    for sub in ['metrics', 'predictions']:
        for f in (sp / folder / sub).glob(f'{old}_*'):
            if not re.fullmatch(rf'{re.escape(old)}_(history|val_.*|test_.*)\.(csv|json)', f.name): continue
            shutil.copy2(f, out / sub / f.name.replace(old, new, 1)); n += 1
for f in ['splits.csv', 'stage2_config.json', 'shortcut_diagnostic.csv', 'exact_duplicates_removed.csv', 'near_duplicate_pairs.csv']:
    shutil.copy2(sp / 'r18orig' / 'metrics' / f, out / 'metrics' / f)
t = pd.read_csv(sp / 's6b' / 'metrics' / 'experiment_table.csv')
o = pd.read_csv(sp / 'r18orig' / 'metrics' / 'experiment_table.csv'); o = o[o.experiment_id == 'R18-C'].copy(); o['experiment_id'] = 'R18-C-orig'
t = pd.concat([t, o], ignore_index=True)
order = ['B1', 'B2', 'R18-A', 'R18-B', 'R18-C-orig', 'R18-C', 'R18-C-s43', 'R18-C-s44', 'T1-lr1e-5', 'T2-lr1e-3', 'T3-bs16', 'T4-adamw', 'T5-dropout', 'T6-weighted', 'T7-hflip', 'R50-A', 'R50-B', 'R50-C', 'F-R50-s43', 'F-R50-s44']
t['o'] = t.experiment_id.map({k: i for i, k in enumerate(order)}); assert t.o.notna().all()
t = t.sort_values('o').drop(columns='o')
notes = {'R18-C-orig': 'Stage 4 execution of seed 42 (first run of this configuration)',
         'R18-C': 'Stage 6 execution of seed 42; used in the 3-seed comparison and as the final checkpoint candidate',
         'R50-C': 'Stage 5 execution of seed 42', 'B1': 'main baseline', 'B2': 'baseline with class-weighted loss; reached the 40-epoch limit'}
t['notes'] = t.experiment_id.map(notes).fillna('')
t.to_csv(out / 'metrics' / 'experiment_table_master.csv', index=False)
print('copied', n, 'files;', len(t), 'runs:', t.experiment_id.tolist())
missing = [e for e in order for s in ['val', 'test'] if not (out / 'predictions' / f'{e}_{s}_predictions.csv').exists()]
print('runs without prediction files:', missing or 'none')
