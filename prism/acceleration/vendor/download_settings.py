"""Model mirror preferences and explicit commands for active download workers."""
import json
import math
from pathlib import Path
import threading
import time
import uuid
from .monitoring import save
from .proxy import MODES

CHOICES = ('auto', 'official', 'hf-mirror', 'modelscope')
FAMILIES = ('edge-models', 'vdn-models', 'models')


def speed_text(row, zh=False):
    if not row.get('ok'):
        return '未连通' if zh else 'Unavailable'
    rate = row.get('bytes_per_second')
    if (row.get('speed_measured') is False or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or rate <= 0):
        return '已连通，样本不足' if zh else 'Connected; insufficient sample'
    speed = rate / 2**20
    return '<0.01 MiB/s' if speed < .005 else (('%.2f' if speed < 1 else '%.1f') % speed + ' MiB/s')


def read(path):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return dict(source='auto', proxy_mode='auto')
    if not isinstance(value,dict) or value.get('source') not in CHOICES:
        raise ValueError('Invalid model download settings: '+str(path))
    value.setdefault('proxy_mode', 'auto')
    if value['proxy_mode'] not in MODES:
        raise ValueError('Invalid download connection mode')
    return value


def preference(root, source=None, *, proxy_mode=None):
    if source is not None and source not in CHOICES:
        raise ValueError('Choose Auto, Hugging Face, HF Mirror or ModelScope')
    if proxy_mode is not None and proxy_mode not in MODES:
        raise ValueError('Choose Auto, Proxy only or Direct only')
    path=Path(root)/'download-settings.json'
    value=read(path)
    if source is not None:
        value['source'] = source
    if proxy_mode is not None:
        value['proxy_mode'] = proxy_mode
    value.update(updated_at=time.time(),switch_id=uuid.uuid4().hex)
    save(path,value)
    return value


class SwitchRequested(Exception):
    """Separate user intent from a failed source; never retry the old source."""
    def __init__(self, value):
        self.value=value
        super().__init__('Download settings changed')


def current(network):
    path=network.get('download_settings_path')
    return read(path) if path else dict(source='auto', proxy_mode=network.get('proxy_mode', 'auto'))


def order(network, family, names):
    if network.get('mode')=='official':
        return ['official']
    path=network.get('download_settings_path')
    if not path:
        return names
    value=read(path)
    rows=value.get('probe',{}).get('sources',{}).get(family,[])
    fresh=(time.time()-value.get('probe',{}).get('measured_at',0) < 86400 and
           value.get('probe',{}).get('proxy_mode', 'auto') == value['proxy_mode'])
    if fresh:
        measured=list(dict.fromkeys(r['id'] for r in rows if r.get('ok') and r.get('id') in names))
        names=measured+[n for n in names if n not in measured]
    selected=value['source']
    # ModelScope has no verified encoder mapping. That family retains its
    # supported HF sources instead of constructing an invalid download URL.
    return [selected]+[n for n in names if n!=selected] if family in FAMILIES and selected in names else names


class Probe:
    def __init__(self):
        self.lock=threading.Lock()
        self.state=dict(status='idle')
        self.thread=None

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def start(self, root, hf_token=''):
        from .hf_auth import environment
        env = environment(hf_token)
        root=Path(root)
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return dict(self.state)
            self.state=dict(status='running',root=str(root))
            selected_mode = read(root/'download-settings.json')['proxy_mode']
            def work():
                try:
                    from . import network
                    from .environments import bootstrap_versions
                    package=Path(__file__).parent
                    spec=json.loads((package/'dependencies.json').read_text())
                    versions=bootstrap_versions(json.loads((package/'bootstrap_versions.json').read_text()))
                    def progress(value):
                        with self.lock:
                            self.state.update(progress=value)
                    result=network.plan(spec,versions,timeout=5,env=env,proxy_mode=selected_mode,
                                        measure_speed=True,progress=progress)
                    fields=('id','route','ok','bytes_per_second','seconds','bytes','speed_measured',
                            'startup_seconds','ttfb_seconds','transfer_seconds','measured_bytes','stop_reason',
                            'score', 'body_stalled')
                    unavailable = not any(r.get('ok') for rows in result['sources'].values() for r in rows)
                    result=dict(measured_at=time.time(),method='http-body-v2',proxy_mode=selected_mode,network_unavailable=unavailable,sources={family:[{k:r[k] for k in fields if k in r}
                        for r in rows] for family,rows in result['sources'].items()})
                    # Keep any preference chosen while probes were in flight.
                    with self.lock:
                        value=read(root/'download-settings.json')
                        if value['proxy_mode'] != selected_mode:
                            self.state=dict(status='idle',root=str(root))
                            return
                        value['probe']=result
                        save(root/'download-settings.json',value)
                        self.state=dict(status='complete',root=str(root),**result)
                except Exception as error:
                    with self.lock:
                        # Provider exceptions can contain request headers/URLs.
                        self.state=dict(status='failed',root=str(root),
                            error='Source speed test failed (' + type(error).__name__ + '). Retry the speed test.')
            self.thread=threading.Thread(target=work,daemon=True)
            self.thread.start()
            return dict(self.state)

    def select(self, root, source=None, *, proxy_mode=None):
        with self.lock:
            return preference(root,source,proxy_mode=proxy_mode)
