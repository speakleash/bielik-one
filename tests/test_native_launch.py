import json
import pytest
from test_launch import dry


@pytest.fixture(autouse=True)
def required_tools(tmp_path,monkeypatch):
    tools=tmp_path/'external-chat-parsers'/'tools';tools.mkdir(parents=True)
    for name in ['bielik_vllm_tool_parser.py','bielik_vllm_reasoning_parser.py']:
        (tools/name).write_text('# external fixture')
    monkeypatch.setenv('BIELIK_TOOLS_DIR',str(tools.parent))


def original(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'llama','architectures':['LlamaForCausalLM']}))
    return str(tmp_path)


def test_native_backend_has_no_adapter_or_plugin(tmp_path):
    result=dry('backend',original(tmp_path),'--native-tokens')
    cmd=result['command']
    assert result['kind']=='native-tokens' and result['checksums_verified'] is False
    assert result['env']['VLLM_PLUGINS']==''
    assert '--enable-lora' not in cmd and '--hf-overrides' not in cmd
    assert cmd[cmd.index('--logprobs-mode')+1]=='raw_logprobs'


def test_native_frontend_explicit_plain_suffix(tmp_path):
    cmd=dry('frontend',original(tmp_path),'--native-tokens','--decision-prefix','plain','--max-num-seqs','12')['command']
    assert 'bielik_one.api.base_tokens' in cmd
    assert cmd[cmd.index('--decision-prefix')+1]=='plain'
    assert cmd[cmd.index('--workers')+1]=='12'


def test_native_profile_refuses_adapter_package(tmp_path):
    root=original(tmp_path);(tmp_path/'heads.json').write_text('{}')
    with pytest.raises(ValueError,match='original weights'):
        dry('backend',root,'--native-tokens')


def test_non_native_rejects_plain_suffix(tmp_path):
    with pytest.raises(ValueError,match='only to --native-tokens'):
        dry('frontend',original(tmp_path),'--decision-prefix','plain')


def test_native_limits_forwarded_and_old_profiles_not_silently_affected(tmp_path):
    root=original(tmp_path)
    cmd=dry('frontend',root,'--native-tokens','--read-queue-size','10','--max-requests','4','--backend-timeout','30','--request-timeout','60')['command']
    assert cmd[cmd.index('--queue-size')+1]=='10' and cmd[cmd.index('--max-requests')+1]=='4'
    assert cmd[cmd.index('--backend-timeout')+1]=='30.0' and cmd[cmd.index('--request-timeout')+1]=='60.0'
    with pytest.raises(ValueError,match='only to'):
        dry('frontend',root,'--request-timeout','60')
    with pytest.raises(ValueError,match='finite and positive'):
        dry('frontend',root,'--native-tokens','--request-timeout','nan')


def test_native_does_not_silently_ignore_newer_engine_flags(tmp_path):
    root=original(tmp_path)
    with pytest.raises(ValueError,match='requires --engine vllm'):
        dry('backend',root,'--native-tokens','--engine','mlx')


def test_native_backend_accepts_opt_in_fp8(tmp_path):
    cmd=dry('backend',original(tmp_path),'--native-tokens','--fp8')['command']
    assert cmd[cmd.index('--quantization')+1]=='fp8_per_tensor'
    assert json.loads(cmd[cmd.index('--quantization-config')+1])=={'ignore':['*lm_head*']}


def test_native_calibration_forwarding_and_disable(tmp_path):
    root=original(tmp_path)
    calibration=tmp_path/'custom.json';calibration.write_text('{}')
    cmd=dry('frontend',root,'--native-tokens','--native-calibration',str(calibration))['command']
    assert cmd[cmd.index('--calibration')+1]==str(calibration)
    cmd=dry('frontend',root,'--native-tokens','--no-calibration')['command']
    assert '--no-calibration' in cmd
    with pytest.raises(ValueError,match='conflicts'):
        dry('frontend',root,'--native-tokens','--native-calibration',str(calibration),'--no-calibration')
    with pytest.raises(ValueError,match='only to frontend'):
        dry('backend',root,'--native-tokens','--native-calibration',str(calibration))


def test_native_frontend_cli_reaches_real_application_startup(tmp_path,monkeypatch):
    import sys,uvicorn
    from transformers import AutoTokenizer
    from fastapi.testclient import TestClient
    from test_base_tokens import Tok
    from bielik_one.api import base_tokens
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**kw:Tok())
    monkeypatch.setattr(sys,'argv',['base_tokens','--base-model',str(tmp_path),'--backend-url','http://backend'])
    started=[]
    def run(app,**kwargs):
        with TestClient(app) as client:
            assert client.get('/openapi.json').status_code==200
        started.append(kwargs)
    monkeypatch.setattr(uvicorn,'run',run)
    base_tokens.main()
    assert started and started[0]['port']==8026


def test_native_chat_parsers_reuse_shipped_template_and_reject_opt_out(tmp_path,monkeypatch):
    import shlex
    from bielik_one import launch
    root=original(tmp_path)
    tools=tmp_path/'bielik tools'/'tools';tools.mkdir(parents=True)
    for name in ['bielik_vllm_tool_parser.py','bielik_vllm_reasoning_parser.py']:
        (tools/name).write_text('# external parser fixture')
    monkeypatch.setenv('BIELIK_TOOLS_DIR','/missing/environment')
    cmd=dry('backend',root,'--native-tokens','--bielik-tools-dir',str(tools.parent))['command']
    assert cmd[cmd.index('--tool-parser-plugin')+1]==str(tools/'bielik_vllm_tool_parser.py')
    assert cmd[cmd.index('--reasoning-parser-plugin')+1]==str(tools/'bielik_vllm_reasoning_parser.py')
    assert cmd[cmd.index('--tool-call-parser')+1]=='bielik'
    assert cmd[cmd.index('--reasoning-parser')+1]=='bielik'
    assert '--enable-auto-tool-choice' in cmd and '--chat-template' not in cmd
    with pytest.raises(ValueError,match='parsers are required'):
        dry('backend',root,'--native-tokens','--no-bielik-chat-parsers')
    monkeypatch.setenv('BIELIK_TOOLS_DIR',str(tools.parent))
    assert dry('backend',root,'--native-tokens')['command']==cmd
    assert '--tool-call-parser' not in dry('frontend',root,'--native-tokens')['command']
    with pytest.raises(ValueError,match='vLLM backend options'):
        dry('frontend',root,'--native-tokens','--bielik-tools-dir',str(tools.parent))
    (tools/'bielik_vllm_tool_parser.py').unlink()
    with pytest.raises(ValueError,match='Missing bielik-tools file'):
        dry('backend',root,'--native-tokens')


def test_native_missing_parser_checkout_fails_before_start(tmp_path,monkeypatch):
    monkeypatch.delenv('BIELIK_TOOLS_DIR')
    with pytest.raises(ValueError,match='parsers are required'):
        dry('backend',original(tmp_path),'--native-tokens')


def test_native_frontend_passes_generation_limits(tmp_path):
    root=original(tmp_path)
    cmd=dry('frontend',root,'--native-tokens')['command']
    assert '--max-generation-n' not in cmd and '--max-generation-tokens' not in cmd
    cmd=dry('frontend',root,'--native-tokens','--max-generation-n','4','--max-generation-tokens','none')['command']
    assert cmd[cmd.index('--max-generation-n')+1]=='4'
    assert cmd[cmd.index('--max-generation-tokens')+1]=='none'


def test_native_frontend_accepts_generation_limits():
    import subprocess,sys
    pytest.importorskip('transformers')
    out=subprocess.run([sys.executable,'-m','bielik_one.api.base_tokens','--help'],capture_output=True,text=True,check=True).stdout
    assert '--max-generation-n' in out and '--max-generation-tokens' in out
