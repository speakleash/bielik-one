"""Original Bielik answer tokens: model-specific prompt and output interpretation."""
from __future__ import annotations

import math
from pathlib import Path
from functools import lru_cache
from bielik_one.api.remote_vllm import RemoteVllmEngine, build_remote_app, resolve_calibration
from bielik_one.api.vllm_reads import (
    VllmReadTransport as VllmTokenReadout, conditional_logprobs, check_cancelled,
)
from bielik_one.prompting.inference import inference_messages
from bielik_one.prompting.letter import answer_token_ids, label_names, label_pool, safe_messages

def decision_ids(tok, row, *, read_twice=False, empty_think=True, **kwargs):
    chat = safe_messages(tok, inference_messages(tok, row, True, read_twice=read_twice, **kwargs))
    text = (tok.bos_token or '') + ''.join(
        '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' for m in chat
    ) + '<|im_start|>assistant\n' + ('<think></think>' if empty_think else '')
    ids = list(tok.encode(text, add_special_tokens=False))
    # Check the real continuation, not an assumed whitespace-dependent token ID.
    symbols = ([o.key for o in row.options] if row.question.type in ('noul', 'score')
               else label_names(tok, len(row.options)))
    wanted = answer_token_ids(tok, row, True)
    for symbol, token in zip(symbols, wanted, strict=True):
        check_answer_token(tok, symbol, token, empty_think)
    return ids


@lru_cache(maxsize=2048)
def check_answer_token(tok, symbol, token, empty_think=True):
    # The final special token fixes the continuation boundary for every prompt.
    suffix = '<|im_start|>assistant\n' + ('<think></think>' if empty_think else '')
    ids = list(tok.encode(suffix, add_special_tokens=False))
    if list(tok.encode(suffix+symbol, add_special_tokens=False)) != ids+[token]:
        raise ValueError(f'{symbol!r} is not the expected single token after the decision prefix')


class BaseTokenEngine(RemoteVllmEngine):
    strict_read_twice_context = True
    report_output_reads = False  # Preserve native response accounting.

    def __init__(self,tok,readout,*,model_name='decision',max_model_len=32768,empty_think=True):
        self.tok,self.readout,self.model_name = tok,readout,model_name
        self.empty_think,self.max_model_len = empty_think,max_model_len
        self.capacity = min(255,len(label_pool(tok)))
        self.labels_capacity=0;self.parking=None;self.labels_threshold=.5;self.semantic=True
        self.temperatures={};self.calibration_source=None;self.reasoners={};self.aliases={model_name}

    def prompt_ids(self,row,**kwargs):
        if kwargs.pop('analysis',''):
            raise ValueError('Reasoning not supported by native base token readout')
        return decision_ids(self.tok,row,empty_think=self.empty_think,**kwargs)

    def per_option_probability(self,tables,token_ids,orders,temperature,yes):
        from bielik_one.api.server import score_orders,apply_temperature
        # The published labels calibration was fitted on the serialized scalar
        # P(yes), then transformed as [1-p, p]. Preserve that fitting contract,
        # including float32 rounding; using the original pair changes the fit.
        prob=score_orders(tables,token_ids,orders,1.)[yes]
        return apply_temperature([1-prob,prob],temperature)[1]

    def per_option_temperature_kind(self,row):
        return "labels" if row.question.type=="labels" else "noul"

    def evaluate(self,body):
        body=body.model_copy(update={'questions':{
            key:q.model_copy(update={'mode':'per_option'}) if q.type=='labels' else q
            for key,q in body.questions.items()}})
        return super().evaluate(body)

    def read_metadata(self,outputs,prompt):
        count=self.readout.read_count(outputs) if isinstance(self.readout,VllmTokenReadout) else 1
        return {'backend_reads':count,'backend_input_tokens':len(prompt)*count}

    def read_tasks(self,tasks):
        values=self.readout([(prompt,outputs) for prompt,outputs,_ in tasks])
        if len(values)!=len(tasks):raise ValueError('Missing backend reads')
        return [(conditional_logprobs(table,outputs),self.read_metadata(outputs,prompt)['backend_input_tokens'])
                for table,(prompt,outputs,_) in zip(values,tasks,strict=True)]

    def close(self):
        if hasattr(self.readout,'close'):self.readout.close()


def main():
    import argparse,uvicorn
    from transformers import AutoTokenizer
    from bielik_one.api.remote_vllm import build_remote_app
    from bielik_one.api.http_guard import add_guard_arguments, guarded
    p=argparse.ArgumentParser(description=__doc__)
    add_guard_arguments(p)
    from bielik_one.api.remote_vllm import add_calibration_arguments, add_generation_arguments, generation_limits
    add_calibration_arguments(p)
    add_generation_arguments(p)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--queue-size',type=int,default=64)
    p.add_argument('--max-requests',type=int,default=32)
    p.add_argument('--backend-timeout',type=float,default=180)
    p.add_argument('--request-timeout',type=float,default=300)
    p.add_argument('--decision-prefix',choices=['empty-think','plain'],default='empty-think')
    p.add_argument('--base-model',required=True);p.add_argument('--backend-url',required=True)
    p.add_argument('--backend-model',default='base');p.add_argument('--model-name',default='decision')
    p.add_argument('--max-model-len',type=int,default=32768);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8026)
    a=p.parse_args()
    if not 1 <= a.workers <= 256: p.error('--workers must be between 1 and 256')
    if not 0 <= a.queue_size <= 4096: p.error('--queue-size must be between 0 and 4096')
    if not 1 <= a.max_requests <= 256: p.error('--max-requests must be between 1 and 256')
    if any(not math.isfinite(v) or v <= 0 for v in (a.backend_timeout,a.request_timeout)): p.error('timeouts must be finite and positive')
    tok=AutoTokenizer.from_pretrained(a.base_model,local_files_only=True)
    engine=BaseTokenEngine(tok,VllmTokenReadout(a.backend_url,a.backend_model,workers=a.workers,queue_size=a.queue_size,timeout=a.backend_timeout),model_name=a.model_name,max_model_len=a.max_model_len,empty_think=a.decision_prefix=='empty-think')
    try:
        if a.calibration is not None and a.no_calibration:
            raise ValueError('Explicit calibration conflicts with disabled calibration')
        engine.temperatures,engine.calibration_source=resolve_calibration(Path(a.base_model)/'config.json',a.calibration,a.no_calibration)
    except (ValueError,OSError) as exc:
        engine.close();p.error(str(exc))
    engine.backend_url=a.backend_url.rstrip('/')
    uvicorn.run(guarded(build_remote_app(engine,chat_model=a.backend_model,error_format=a.error_format,generation_limits=generation_limits(a,a.max_model_len),max_requests=a.max_requests,request_timeout=a.request_timeout),a.max_body_mb,a.error_format),host=a.host,port=a.port)


if __name__=='__main__':main()
