"""Exercise actual request propagation and backend prompt choice without GPU weights."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from pydantic import ValidationError
from bielik_one.api.remote_vllm import RemoteVllmEngine
from bielik_one.api.server import RequestError, SystemOneIn
import test_remote_vllm as fixtures
import math


def body(**kwargs):
    return SystemOneIn(state='state',model='decision',questions={'q':{'type':'noul','instructions':'Is this correct?'}},**kwargs)


def test_schema_defaults_and_explicit_disable():
    assert body().read_twice is True
    assert body().read_twice_max_input_tokens==4096
    assert body(read_twice=False).read_twice is False
    assert body(read_twice_max_input_tokens=None).read_twice_max_input_tokens is None
    for value in (-1,True,1.5,'4096'):
        with pytest.raises(ValidationError):body(read_twice_max_input_tokens=value)


@pytest.mark.parametrize('standard,repeated,context,threshold,enabled,expected',[
    (4096,8200,10000,4096,True,8200),
    (4097,8202,10000,4096,True,4097),
    (100,199,200,None,True,199),
    (100,200,200,None,True,100),
    (100,199,200,4096,False,100),
    (5000,10000,12000,None,True,10000),
    (100,199,200,0,True,100),
])
def test_actual_backend_prompt(standard,repeated,context,threshold,enabled,expected):
    fixture=fixtures.RemoteVllmTests();engine=fixture.make_engine();engine.max_model_len=context
    row=SimpleNamespace(id='q',question=SimpleNamespace(type='noul'),options=[0,1])
    def prompt(*args,**kw):return [1]*(repeated if kw['read_twice'] else standard)
    try:
        with patch('bielik_one.api.remote_vllm.inference_ids',side_effect=prompt),patch('bielik_one.api.remote_vllm.answer_token_ids',return_value=[1000,1001]):
            _,tokens=engine.distributions([row],read_twice=enabled,read_twice_max_input_tokens=threshold)
        assert len(fixture.requests)==1
        assert len(fixture.requests[0]['prompt'])==expected
        assert tokens==expected
    finally:engine.close()


def test_standard_overflow_never_calls_backend():
    fixture=fixtures.RemoteVllmTests();engine=fixture.make_engine();engine.max_model_len=100
    row=SimpleNamespace(id='q',question=SimpleNamespace(type='noul'),options=[0,1])
    try:
        with patch('bielik_one.api.remote_vllm.inference_ids',return_value=[1]*100):
            with pytest.raises(RequestError):engine.distributions([row],read_twice=True)
        assert fixture.requests==[]
    finally:engine.close()


def test_request_propagates_threshold():
    fixture=fixtures.RemoteVllmTests();engine=fixture.make_engine();seen={}
    def distributions(prompts,**kw):
        seen.update(kw)
        return [{1000:math.log(.5),1001:math.log(.5)}],10
    engine.distributions=distributions
    try:
        with patch('bielik_one.api.server.answer_token_ids',return_value=[1000,1001]):
            engine.evaluate(body(read_twice_max_input_tokens=1234))
        assert seen['read_twice'] is True
        assert seen['read_twice_max_input_tokens']==1234
    finally:engine.close()
