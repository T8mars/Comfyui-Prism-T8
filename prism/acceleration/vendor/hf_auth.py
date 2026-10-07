"""Optional session credentials; never written to a settings or receipt file."""
import os
import re

TOKEN_URL = 'https://huggingface.co/settings/tokens'
HELP_EN = ('Optional: add a Hugging Face read token to use your account’s download quota. '
           'This can reduce anonymous throttling; it does not guarantee higher speed. '
           'Kept for this session only. You can continue without it.')
HELP_ZH = ('可选：填写 Hugging Face 读取 Token，使用账号下载额度，减少匿名限流。'
           '可能更快，但不保证提速。仅本次打开有效，不填写也能继续。')


def validate(value):
    if not isinstance(value, str):
        raise ValueError('Enter a Hugging Face token beginning with hf_, or leave it empty.')
    value = value.strip()
    if value and not re.fullmatch(r'hf_[A-Za-z0-9]{1,4093}', value):
        raise ValueError('Enter a Hugging Face token beginning with hf_, or leave it empty.')
    return value


def environment(token='', environ=None):
    env = dict(os.environ if environ is None else environ)
    token = validate(token)
    if token:
        env['HF_TOKEN'] = token
    return env
