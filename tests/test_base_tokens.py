import math,re
import httpx,pytest
from bielik_one.api.base_tokens import BaseTokenEngine, VllmTokenReadout, decision_ids, conditional_logprobs
from bielik_one.api.server import SystemOneIn,QuestionIn,question_to_row,RequestError

class Tok:
    bos_token='<s>';all_special_tokens=['<s>','<|im_start|>','<|im_end|>','<think>','</think>']
    def encode(self,text,**kwargs):
        parts=re.findall(r'<\|im_start\|>|<\|im_end\|>|<think>|</think>|<s>|yes|no|[\s\S]',text)
        words={w:100+i for i,w in enumerate([*self.all_special_tokens,'yes','no'])}
        return [words[p] if p in words else ord(p)+1000 for p in parts]

def row(kind='choice'):
    c={'choice':{'x':'apple','y':'pear'},'noul':{'true':'positive','false':'negative'},'score':['low','high']}[kind]
    return question_to_row('Some state','q',QuestionIn(type=kind,instructions='Decide',criteria=c))[0]

def test_prompt_contract_and_read_twice():
    t=Tok();r=row();a=decision_ids(t,r);b=decision_ids(t,r,read_twice=True)
    assert a[-1]==104 and b[-1]==104
    assert len(b)>len(a)
    assert a.count(101)==3 and b.count(101)==3  # Exactly system/user/assistant ChatML.
    assert a.count(103)==1 and b.count(103)==1

@pytest.mark.parametrize('kind',['choice','noul','score'])
def test_native_token_distributions(kind):
    t=Tok();seen=[]
    def backend(tasks):
        seen.extend(tasks);return [{i:1000+j for j,i in enumerate(wanted)} for _,wanted in tasks]
    e=BaseTokenEngine(t,backend)
    q=row(kind).question
    b=SystemOneIn(state='State',model='decision',read_twice=False,questions={'q':QuestionIn(type=kind,instructions=q.instructions,criteria=q.criteria)})
    p=e.evaluate(b)['answers']['q']['probabilities']
    assert sum(p.values())==pytest.approx(1) and max(p.values())==pytest.approx(1/(1+math.exp(-1)))
    assert len(seen)==1

def test_labels_are_only_noul_without_head():
    seen=[]
    def backend(tasks):seen.extend(tasks);return [{i:0 for i in wanted} for _,wanted in tasks]
    e=BaseTokenEngine(Tok(),backend)
    b=SystemOneIn(state='s',model='decision',read_twice=False,questions={'q':QuestionIn(type='labels',instructions='Select all',criteria={'x':'apple','y':'pear','z':'banana'})})
    result=e.evaluate(b)['answers']['q']
    assert result['mode']=='per_option' and result['probabilities']=={'x':.5,'y':.5,'z':.5}
    assert len(seen)==3 and all(len(ids)==2 for _,ids in seen)
    assert e.parking is None

def test_context_rejection_before_model_call():
    e=BaseTokenEngine(Tok(),lambda _:pytest.fail('must not call GPU'),max_model_len=5)
    with pytest.raises(RequestError,match='context limit'):
        e.distributions([row()],read_twice=True)

def test_native_prompt_details_count_actual_per_option_and_permutation_reads():
    seen=[]
    def backend(tasks):
        seen.extend(tasks)
        return [{token:0 for token in wanted} for _,wanted in tasks]
    engine=BaseTokenEngine(Tok(),backend)
    body=SystemOneIn(model='decision',state='A pear is present.',read_twice=True,
        read_twice_max_input_tokens=None,prompt_details=True,permutations=2,questions={
            'pick':QuestionIn(type='choice',instructions='Pick a fruit.',criteria={'x':'apple','y':'pear'}),
            'tags':QuestionIn(type='labels',instructions='Select all fruits.',criteria={'x':'apple','y':'pear'})})
    result=engine.evaluate(body)
    for name,count in [('pick',2),('tags',4)]:
        prompt=result['answers'][name]['prompt']
        assert prompt['variant']=='read_twice' and prompt['reason']=='applied'
        assert prompt['read_twice_reads']==count and prompt['standard_reads']==0
        assert len(prompt['reads'])==count
        assert all(r['input_tokens']>r['standard_input_tokens'] for r in prompt['reads'])
    assert result['usage']['input_tokens']==sum(len(ids) for ids,_ in seen)
    assert result['usage']['input_tokens']==sum(r['input_tokens'] for a in result['answers'].values() for r in a['prompt']['reads'])
    standard=engine.evaluate(body.model_copy(update={'read_twice':False,'prompt_details':False}))
    assert all(a['prompt']['variant']=='standard' and a['prompt']['reason']=='disabled' and 'reads' not in a['prompt'] for a in standard['answers'].values())

def test_native_read_twice_threshold_applies_to_whole_question_not_individual_reads():
    engine=BaseTokenEngine(Tok(),lambda tasks:[{token:0 for token in wanted} for _,wanted in tasks])
    body=SystemOneIn(model='decision',state='s',read_twice=False,prompt_details=True,questions={
        'tags':QuestionIn(type='labels',instructions='Select all.',criteria={'x':'apple','y':'a much longer candidate '*10}),
        'other':QuestionIn(type='noul',instructions='Present?',criteria={'true':'yes','false':'no'})})
    standard=engine.evaluate(body)
    lengths=[r['input_tokens'] for r in standard['answers']['tags']['prompt']['reads']]
    assert min(lengths)<max(lengths)
    result=engine.evaluate(body.model_copy(update={'read_twice':True,'read_twice_max_input_tokens':min(lengths)}))
    tags=result['answers']['tags']['prompt'];other=result['answers']['other']['prompt']
    assert tags['variant']=='standard' and tags['reason']=='input_threshold'
    assert tags['standard_reads']==2 and tags['read_twice_reads']==0
    assert all(r['input_tokens']==r['standard_input_tokens'] for r in tags['reads'])
    assert other['variant']=='read_twice' and other['read_twice_reads']==1




@pytest.mark.parametrize('kind',['choice','noul','score'])
def test_no_think_prefix_preserves_answer_tokens(kind):
    t=Tok();r=row(kind)
    ids=decision_ids(t,r,empty_think=False)
    assert 103 not in ids and 104 not in ids
    assert ids[-1]==ord('\n')+1000
    assert len(decision_ids(t,r,empty_think=False,read_twice=True))>len(ids)










def test_native_http_endpoint_and_context_error():
    from fastapi.testclient import TestClient
    from bielik_one.api.remote_vllm import build_remote_app
    backend=lambda tasks:[{i:0 for i in wanted} for _,wanted in tasks]
    engine=BaseTokenEngine(Tok(),backend);engine.backend_url='http://backend'
    with TestClient(build_remote_app(engine,max_requests=32)) as client:
        body={'model':'decision','state':'s','read_twice':False,'questions':{'q':{'type':'noul','instructions':'Is this correct?'}}}
        r=client.post('/v1/systemone',json=body)
        assert r.status_code==200 and r.json()['answers']['q']['probabilities']=={'false':.5,'true':.5}
        engine.max_model_len=1
        assert client.post('/v1/systemone',json=body).status_code==422








def test_native_facts_keep_shared_context_fallback(monkeypatch):
    from bielik_one.api import server
    monkeypatch.setattr(server,'compute_facts',lambda state,instructions:(['x'*2000],'en'))
    calls=[]
    def read(tasks):calls.extend(tasks);return [{i:0 for i in w} for _,w in tasks]
    tok=Tok();q=QuestionIn(type='noul',instructions='Is this correct?')
    r=question_to_row('state','q',q)[0]
    e=BaseTokenEngine(tok,read,max_model_len=len(decision_ids(tok,r))+1)
    result=e.evaluate(SystemOneIn(model='decision',state='state',read_twice=False,facts=True,questions={'q':q}))
    assert result['facts']['lines']==[] and len(calls)==1


def test_native_preserves_canonical_first_row_for_semantic_read_twice(monkeypatch):
    from bielik_one.api import base_tokens
    from bielik_one.prompting.orders import _reordered
    canonical=row('noul');reverse=_reordered(canonical,[1,0]);seen=[]
    original=base_tokens.decision_ids
    def capture(tok,r,**kwargs):seen.append(kwargs);return original(tok,r,**kwargs)
    monkeypatch.setattr(base_tokens,'decision_ids',capture)
    engine=BaseTokenEngine(Tok(),lambda tasks:[{i:0 for i in w} for _,w in tasks])
    engine.distributions([reverse],first_rows=[canonical],read_twice=True)
    assert all(kw['first_row'] is canonical for kw in seen)
    with pytest.raises(ValueError,match='original row'):
        engine.distributions([reverse],first_rows=[canonical,canonical])


def test_labels_count_is_not_limited_by_choice_token_pool():
    import json
    calls=[]
    def respond(request):
        d=json.loads(request.content);calls.append(d)
        return httpx.Response(200,json={'usage':{'prompt_tokens':len(d['prompt'])},'choices':[{'logprobs':{'top_logprobs':[{f'token_id:{i}':-1 for i in d['logprob_token_ids']}]}}]})
    reader=VllmTokenReadout('http://backend','base',workers=4,client=httpx.Client(transport=httpx.MockTransport(respond)))
    engine=BaseTokenEngine(Tok(),reader)
    try:
        body=SystemOneIn(model='decision',state='s',read_twice=False,questions={'q':QuestionIn(type='labels',instructions='Select all',criteria={f'k{i}':str(i) for i in range(129)})})
        result=engine.evaluate(body)['answers']['q']
        assert len(result['probabilities'])==129 and len(calls)==129
        assert all(len(c['logprob_token_ids'])==2 for c in calls)
        calls.clear()
        with pytest.raises(RequestError,match='checkpoint supports at most'):
            engine.evaluate(body.model_copy(update={'questions':{'q':QuestionIn(type='choice',instructions='Pick one',criteria={f'k{i}':str(i) for i in range(129)})}}))
        assert calls==[]
    finally:reader.close()


def test_native_labels_use_separate_calibration_from_noul():
    def backend(tasks):
        return [{token:float(i) for i,token in enumerate(wanted)} for _,wanted in tasks]
    engine=BaseTokenEngine(Tok(),backend)
    engine.temperatures={'1':{'labels':2,'noul':1}}
    body=SystemOneIn(model='decision',state='s',read_twice=True,read_twice_max_input_tokens=None,
        questions={'tags':QuestionIn(type='labels',instructions='Select',criteria={'a':'apple','b':'pear'}),
                   'binary':QuestionIn(type='noul',instructions='Present?')})
    result=engine.evaluate(body)['answers']
    assert result['tags']['probabilities']['a']==pytest.approx(1/(1+math.exp(-.5)))
    assert result['binary']['noul']==pytest.approx(1/(1+math.exp(-1)))


def test_native_calibration_applies_across_request_variants(tmp_path):
    import json
    path=tmp_path/'calibration.json'
    value={'temperatures':{'choice':2,'binary':3,'score':4,'labels':5},
           'configuration':{'read_twice':True,'read_twice_max_input_tokens':None,
                            'permutations':1,'decision_prefix':'empty-think','precision':'bf16'}}
    path.write_text(json.dumps(value))
    def backend(tasks):
        return [{token:float(i) for i,token in enumerate(wanted)} for _,wanted in tasks]
    for prefix in [True,False]:
        engine=BaseTokenEngine(Tok(),backend,empty_think=prefix)
        engine.load_calibration(path)
        assert engine.temperatures['1']['labels']==5
        for rt,permutations,qbs in [(False,1,False),(True,1,False),(True,2,False),(False,1,True)]:
            result=engine.evaluate(SystemOneIn(model='decision',state='s',read_twice=rt,
                permutations=permutations,questions_before_state=qbs,
                questions={'q':QuestionIn(type='noul',instructions='Present?')}))
            assert result['answers']['q']['type']=='noul'
    value['complete_validation_collection']=False;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='provisional'):
        engine.load_calibration(path)


def test_native_labels_temperature_preserves_scalar_calibration_contract():
    from bielik_one.api.server import apply_temperature,score_orders
    engine=BaseTokenEngine(Tok(),lambda tasks:[])
    from bielik_one.api.vllm_reads import conditional_logprobs
    tables=[conditional_logprobs({7:0.,8:20.},[7,8])]
    ids=[7,8];orders=[[0,1]]
    raw=score_orders(tables,ids,orders,1.)[1]
    assert raw==1.  # Saved scalar predictions lose the tiny P(no).
    actual=engine.per_option_probability(tables,ids,orders,10.,1)
    assert actual==pytest.approx(apply_temperature([1-raw,raw],10.)[1])
    assert actual!=pytest.approx(score_orders(tables,ids,orders,10.)[1])


@pytest.mark.parametrize('options,reads',[(2,1),(127,1),(128,1),(129,2),(255,2),(256,3)])
def test_native_request_token_limit_counts_reads_like_the_transport(options,reads):
    from types import SimpleNamespace
    from bielik_one.api.remote_vllm import RemoteVllmEngine
    choice=SimpleNamespace(options=[str(i) for i in range(options)],question=SimpleNamespace(type='choice'))
    assert BaseTokenEngine(Tok(),None)._reads_per_prompt(choice)==reads
    assert reads==VllmTokenReadout.read_count(list(range(options)))
    # Adapter and merged engines send the anchor beside up to 127 options: 128 options take two reads there.
    adapter=RemoteVllmEngine.__new__(RemoteVllmEngine);adapter.base_row_types=set()
    assert adapter._reads_per_prompt(choice)==-(-options//127)


def test_native_request_token_limit_does_not_overcount_128_options(monkeypatch):
    import bielik_one.api.remote_vllm as remote
    import bielik_one.api.server as server
    letters=lambda tok,row,semantic:list(range(1000,1000+len(row.options)))
    monkeypatch.setattr(remote,'answer_token_ids',letters);monkeypatch.setattr(server,'answer_token_ids',letters)
    engine=BaseTokenEngine(Tok(),lambda tasks:[{i:0. for i in wanted} for _,wanted in tasks])
    engine.capacity=255;engine.max_request_tokens=75
    monkeypatch.setattr(engine,'prompt_ids',lambda row,**kwargs:[7]*50)
    def body(n):
        return SystemOneIn(model='decision',state='s',read_twice=False,questions={
            'q':QuestionIn(type='choice',instructions='Pick',criteria={f'k{i}':str(i) for i in range(n)})})
    assert len(engine.evaluate(body(128))['answers']['q']['probabilities'])==128  # one read of 50 tokens
    with pytest.raises(RequestError,match='more than 75 prompt tokens'):
        engine.evaluate(body(129))  # two reads, 100 tokens
