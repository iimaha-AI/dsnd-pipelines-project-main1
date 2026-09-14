"""Small real-data fit and serialization check; not a model quality benchmark."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'starter'))
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

nb = json.loads((ROOT / 'starter/starter.ipynb').read_text(encoding='utf-8'))
scope = {'display': lambda *args: None}
# Execute original imports, dataset loading, transformer import and pipeline definition.
for index in [5, 7, 24, 26]:
    exec(compile(''.join(nb['cells'][index]['source']), f'notebook-cell-{index+1}', 'exec'), scope)
df = scope['df']
sample, _ = train_test_split(df, train_size=512, random_state=42, stratify=df['Recommended IND'])
X = sample.drop(columns=['Recommended IND'])
y = sample['Recommended IND']
# Limit parallelism for constrained CI/sandbox runners; estimator settings are unchanged.
model = scope['pipeline'].set_params(classifier__n_jobs=1).fit(X, y)
expected = model.predict(X.iloc[:3]).tolist()
with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory)
    joblib.dump(model, folder / 'smoke.pkl')
    X.iloc[:3].to_json(folder / 'inputs.json')
    code = "import joblib,pandas as pd,sys,json; from pathlib import Path; p=Path(sys.argv[1]); print(json.dumps(joblib.load(p/'smoke.pkl').predict(pd.read_json(p/'inputs.json')).tolist()))"
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'starter') + os.pathsep + env.get('PYTHONPATH', '')
    actual = json.loads(subprocess.check_output([sys.executable, '-c', code, str(folder)], env=env, text=True))
    assert actual == expected, (actual, expected)
print('PASS: 512 real rows fitted; predictions survive reload in a fresh process. Full grid search not run.')
