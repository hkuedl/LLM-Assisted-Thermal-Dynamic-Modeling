"""Generate priors only on explicit --run. Default invocation is a call-count preview."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import time
import config as cfg
import prompt


def api_key(environment_name):
    """Read a provider key from the environment or the ignored local key file."""
    key = os.environ.get(environment_name, '').strip()
    if key:
        return key
    path = Path(__file__).with_name('api_key_local.txt')
    if path.is_file():
        key = path.read_text(encoding='utf-8').strip()
    if not key:
        raise ValueError('Set %s or create api_key_local.txt' % environment_name)
    return key


def input_groups(s):
    rows=[r for r in cfg.read_index(s['input_index']) if cfg.selected(r,s)]
    groups=[];seen=set()
    for row in rows:
        building,ood=row.get('building',row.get('Building')),row.get('ood',row.get('OOD_Label'))
        if (building,ood) in seen:raise ValueError('Duplicate generation group: '+str((building,ood)))
        seen.add((building,ood))
        control=cfg.resolve_input(row['control_prior_json'],s['input_index'],s['final_root'])
        evidence=cfg.resolve_input(row['event_evidence_json'],s['input_index'],s['final_root'])
        request=prompt.build_request(cfg.read_json(control),cfg.read_json(evidence),building,ood)
        request['batches']=prompt.batches(request,s['zones_per_call'],s['prompt_style'])
        groups.append(request)
    if not groups:raise ValueError('No selected generation groups in '+str(s['input_index']))
    return groups


def cache_path(s,request,sample,batch):
    return (cfg.generation_root(s)/'requests'/request['building']/request['ood']
        /('sample_%03d'%sample)/('event_%03d_batch_%03d.json'%(batch['event_id'],batch['batch'])))


def request_spec(s,batch):
    return {'model':s['model'],'temperature':s['temperature'],'provider':s['provider'],
        'base_url':os.environ.get('OPENAI_BASE_URL','') if s['provider']=='openai' else '',
        'messages':batch['messages']}


def cached_result(s,path,batch):
    saved=cfg.read_json(path)
    if saved.get('request')!=request_spec(s,batch):
        raise ValueError('Cached request differs: '+str(path)+'\nUse a different run_root for changed prompt/model/evidence/batching.')
    return prompt.parse_and_validate(saved['raw_response'],batch['expected'])


def preview(s):
    groups=input_groups(s);rows=[]
    for request in groups:
        total=s['samples']*len(request['batches']);cached=0
        for sample in range(1,s['samples']+1):
            for batch in request['batches']:
                path=cache_path(s,request,sample,batch)
                if path.is_file():
                    cached_result(s,path,batch);cached+=1
        row={'Building':request['building'],'OOD':request['ood'],'Samples':s['samples'],
            'Batches_per_sample':len(request['batches']),'Cached':cached,'New_calls':total-cached}
        rows.append(row);print(row)
    print('Planned new successful batch calls:',sum(r['New_calls'] for r in rows))
    print('Retries are failed attempts, not votes. No API is called by preview.')
    return groups,rows


def provider_client(s):
    if s['provider']=='together':
        from together import Together
        return Together(api_key=api_key('TOGETHER_API_KEY'),max_retries=0)
    from openai import OpenAI
    return OpenAI(api_key=api_key('OPENAI_API_KEY'),
        base_url=os.environ.get('OPENAI_BASE_URL') or None,max_retries=0)


def run(s, execute=False):
    groups,rows=preview(s)
    print('Prompt style:',s['prompt_style'],'; generated outputs:',cfg.generation_root(s))
    if not execute:return rows
    def sample_job(request,sample):
        merged={z:{} for z in request['expected']};client=None
        for batch in request['batches']:
            path=cache_path(s,request,sample,batch)
            if path.is_file():
                result=cached_result(s,path,batch)
            else:
                if client is None:client=provider_client(s)
                for attempt in range(1,s['max_attempts']+1):
                    try:
                        response=client.chat.completions.create(model=s['model'],messages=batch['messages'],
                            temperature=s['temperature'],response_format={'type':'json_object'})
                        content=response.choices[0].message.content or ''
                        result=prompt.parse_and_validate(content,batch['expected'])
                        cfg.write_json(path,{'request':request_spec(s,batch),'raw_response':content})
                        break
                    except Exception as exc:
                        print('FAILED',request['building'],request['ood'],'sample',sample,
                              'event',batch['event_id'],'attempt',attempt,type(exc).__name__,flush=True)
                        # Every failed execution is recorded; no credentials are logged.
                        log=path.with_name(path.stem+'_failure_%d_%d.json'%(time.time_ns(),attempt))
                        cfg.write_json(log,{'error_type':type(exc).__name__,'attempt':attempt,
                            'sample':sample,'event_id':batch['event_id']})
                        if attempt==s['max_attempts']:raise
                        time.sleep(min(attempt,2))
            for zone,events in result.items():merged[zone].update(events)
        print('SAMPLE READY',request['building'],request['ood'],sample,flush=True)
        return merged
    complete={};failures=[]
    with ThreadPoolExecutor(max_workers=s['max_workers']) as pool:
        jobs={pool.submit(sample_job,r,i):(r,i) for r in groups for i in range(1,s['samples']+1)}
        for future in as_completed(jobs):
            request,sample=jobs[future];key=(request['building'],request['ood'])
            try:complete.setdefault(key,{})[sample]=future.result()
            except Exception as exc:failures.append({'Building':key[0],'OOD_Label':key[1],'Sample':sample,'Error_type':type(exc).__name__})
    generated=dict(s,prior_source='generated')
    for request in groups:
        key=(request['building'],request['ood']);samples=complete.get(key,{})
        if len(samples)!=s['samples']:
            print('INCOMPLETE',key,len(samples),'/',s['samples']);continue
        prior=prompt.aggregate(request,[samples[i] for i in range(1,s['samples']+1)],
            {'model':s['model'],'temperature':s['temperature'],'prompt':'original integrated event+phase',
             'provider':s['provider'],'prompt_style':s['prompt_style'],'batching':'event_then_zones','zones_per_call':s['zones_per_call']})
        path=cfg.prior_path(generated,*key)
        unchanged=False
        if path.is_file():
            try:
                unchanged=cfg.read_json(path)==prior
            except ValueError:
                # This prior can be reconstructed from the validated samples above.
                # Retain a malformed previous file before replacing it.
                backup=path.with_name(path.name+'.invalid_'+str(time.time_ns()))
                path.replace(backup)
                print('REBUILD PRIOR FROM SAVED RESPONSES',path,'; previous file:',backup,flush=True)
        if not unchanged:
            cfg.write_json(path,prior)
        # Case-cache signatures include prior mtime. An identical prior must retain
        # its mtime or restarting would unnecessarily invalidate offline results.
        print('PRIOR READY',path,'(unchanged; reused)' if unchanged else '')
    cfg.write_json(cfg.generation_root(s)/'generation_status.json',{'requested_samples':s['samples'],
        'complete_groups':sum(len(complete.get((r['building'],r['ood']),{}))==s['samples'] for r in groups),
        'planned_groups':len(groups),'failed_samples':failures})
    if failures:raise RuntimeError('Some samples failed; valid batch caches remain. Rerun the same command to resume.')
    return rows


if __name__=='__main__':
    settings,execute,_=cfg.cli()
    run(settings,execute)
