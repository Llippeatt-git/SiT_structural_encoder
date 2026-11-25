'''
Persistent experiment log.

Every training run (reconstruction_pipeline_v2.py) and every test/export run
(recon_save_as_gifti.py) appends ONE JSON object per line to experiment_log.jsonl.
Records are appended even when a run crashes or is killed by SLURM (status is then
"failed" or "interrupted"), so the log shows every attempt, not only successes.

Default location: /ceph/chpc/shared/janine_bijsterbosch/lucas_scratch/experiment_log.jsonl. Override with
`logging.experiment_log: /some/path.jsonl` in the config.

Read it back with:
    import pandas as pd
    df = pd.json_normalize(pd.read_json('experiment_log.jsonl', lines=True).to_dict('records'))
'''

import json
import math
import os
import signal
import socket
from datetime import datetime

try:
    import fcntl  # POSIX only; used to stop concurrent SLURM jobs interleaving lines
except ImportError:
    fcntl = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_PATH = os.path.join(REPO_ROOT, 'experiment_log.jsonl')


def _clean(value):
    '''Make numpy/torch scalars and NaN/inf JSON-safe (NaN -> null, which is valid JSON).'''
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, 'item') and callable(value.item):  # numpy / torch scalar
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def hparam_summary(config):
    '''The hyperparameters worth comparing across runs (full config stays in hparams.yml).'''
    tr = config.get('training', {}) or {}
    tf = config.get('transformer', {}) or {}
    sched = config.get('scheduler', {}) or {}
    optimiser = (config.get('optimisation', {}) or {}).get('optimiser')
    return {
        'LR': tr.get('LR'),
        'bs': tr.get('bs'),
        'epochs': tr.get('epochs'),
        'loss': tr.get('loss'),
        'seed': tr.get('seed'),
        'grad_clip': tr.get('grad_clip'),
        'demeaned_flag': tr.get('demeaned'),  # NB: not currently applied in get_dataloaders
        'optimiser': optimiser,
        'weight_decay': (config.get(optimiser, {}) or {}).get('weight_decay') if optimiser else None,
        'scheduler': sched.get('name'),
        'warmup_epochs': sched.get('warmup_epochs'),
        'min_lr': sched.get('min_lr'),
        'dim': tf.get('dim'),
        'depth': tf.get('depth'),
        'dec_depth': tf.get('dec_depth'),
        'heads': tf.get('heads'),
        'mlp_dim': tf.get('mlp_dim'),
        'latent_dim': tf.get('latent_dim'),
        'pool': tf.get('pool'),
        'dropout': tf.get('dropout'),
        'ico_res': (config.get('mesh_resolution', {}) or {}).get('ico_res'),
        'ico_down_res': (config.get('mesh_resolution', {}) or {}).get('ico_down_res'),
        'data_path': (config.get('data', {}) or {}).get('path_to_workdir'),
    }


def log_experiment(record, log_path=None):
    '''
    Append one record as a single JSON line. Never raises: a logging failure must not
    hide the real result or error of the run, so problems are only printed.
    '''
    log_path = log_path or DEFAULT_LOG_PATH
    full = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'host': socket.gethostname(),
    }
    full.update(record)

    try:
        line = json.dumps(_clean(full), allow_nan=False) + '\n'
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, 'a') as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)
        print(f'Experiment record appended to {log_path}')
    except Exception as e:  # noqa: BLE001
        print(f'WARNING: could not write experiment log to {log_path}: {type(e).__name__}: {e}')


def raise_on_sigterm():
    '''
    SLURM sends SIGTERM when a job hits its time limit or is scancel-ed. By default Python
    dies immediately; turning it into SystemExit lets `finally` blocks write the log record.
    '''
    def _handler(signum, frame):
        raise SystemExit(f'received signal {signum} (SLURM time limit or scancel?)')
    signal.signal(signal.SIGTERM, _handler)


def status_from_exception(exc):
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return 'interrupted'
    return 'failed'
