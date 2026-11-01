"""Offline regression tests. Provider calls below are local fakes, never API calls."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import tempfile
import json
import hashlib
from types import SimpleNamespace
import threading
import numpy as np
import pandas as pd
import config as cfg
import prompt
import generate_priors as generation
import apply_priors as application
import visualize_results as visualization

REFERENCE=json.loads((Path(__file__).with_name('reference.json')).read_text())


def setup_data(folder):
    root=Path(folder)/'Final';data=root/'Data';data.mkdir(parents=True)
    (data/'control.json').write_text(json.dumps(REFERENCE['control']))
    (data/'evidence.json').write_text(json.dumps(REFERENCE['neutral_evidence']))
    pd.DataFrame({'T_pred__Z':REFERENCE['raw']}).to_csv(data/'raw.csv',index=False)
    pd.DataFrame({'T_true__Z':np.linspace(21,25,96)}).to_csv(data/'target.csv',index=False)
    case={'Architecture':'MLP','Method':'FiLM','Model_Tag':'model','Seed':'42','Building':'B',
        'Train_Days':'Full','OOD_Label':'OOD-4','External_Variant':'4','Raw_CSV':'Data/raw.csv',
        'Target_CSV':'Data/target.csv','Control_Prior_JSON':'Data/control.json'}
    index=root/'OOD/postprocessing';index.mkdir(parents=True)
    pd.DataFrame([case]).to_csv(index/'case_manifest.csv',index=False)
    pd.DataFrame([{'building':'B','ood':'OOD-4','control_prior_json':'Data/control.json','event_evidence_json':'Data/evidence.json'}]).to_csv(index/'run_manifest.csv',index=False)
    s=cfg.settings(final_root=root,existing_priors=root/'old_priors',buildings=['B'],
        oods=['OOD-4'],train_days=['Full'],samples=3,max_workers=2,
        prompt_style='original',prior_source='existing')
    prior=cfg.prior_path(s,'B','OOD-4');prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps(REFERENCE['prior']))
    return s


class FakeProvider:
    def __init__(self,fail_first=0):
        self.count=0;self.fail_first=fail_first;self.lock=threading.Lock()
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kwargs):
        with self.lock:
            self.count+=1;count=self.count
        if count<=self.fail_first:return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{bad'))])
        cards=json.loads(kwargs['messages'][1]['content'].split('\n\n',1)[1]);out=[]
        for card in cards:
            events=[]
            for ev in card['event_context']:
                events.append({'event_id':ev['event_id'],'event_gate':.6,'confidence':'medium','reason':'local fixture',
                    'phase_adjustment_records':[{'phase_id':p,'phase_adjustment':.02,'reason':'local fixture'} for p in ev['requested_phase_ids']]})
            out.append({'zone':card['zone'],'event_gate_records':events})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'zone_gate_records':out})))])


class PipelineTests(unittest.TestCase):
    def test_server_result_path_remaps_after_release_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'Result'; target=root/'nested/input.csv'
            target.parent.mkdir(parents=True);target.write_text('x\n1\n')
            old='/media/server/Building_Temperature_Final_Results/nested/input.csv'
            resolved=cfg.resolve_input(old,root/'OOD/postprocessing/case_manifest.csv',root)
            self.assertEqual(resolved,target.resolve())

    def test_original_prompt_and_cards(self):
        self.assertEqual(hashlib.sha256(prompt.SYSTEM_PROMPT.encode()).hexdigest(),REFERENCE['prompt_sha256'])
        req=prompt.build_request(REFERENCE['control'],REFERENCE['evidence'],'B','OOD-4')
        self.assertEqual(req['cards'],REFERENCE['expected_cards'])
        neutral=prompt.build_request(REFERENCE['control'],REFERENCE['neutral_evidence'],'B','OOD-4')
        self.assertEqual(req,neutral)
        self.assertNotIn('correction_gate',json.dumps(req['cards']))
        self.assertIn('v5_event_summary',json.dumps(req['cards']))

    def test_public_style_only_relabels_text_and_evidence_key(self):
        req=prompt.build_request(REFERENCE['control'],REFERENCE['evidence'],'B','OOD-4')
        before=json.dumps(req,sort_keys=True)
        original=prompt.messages(req['cards'],'original')
        public=prompt.messages(req['cards'],'public')
        self.assertNotIn('v5',json.dumps(public).lower())
        pub_cards=json.loads(public[1]['content'].split('\n\n',1)[1])
        for card in pub_cards:
            for event in card['event_context']:
                event['v5_event_summary']=event.pop('event_summary')
        self.assertEqual(pub_cards,req['cards'])
        self.assertEqual(json.dumps(req,sort_keys=True),before)
        expected_system=(prompt.SYSTEM_PROMPT.replace('v5_event_summary','event_summary')
            .replace('original V5 ','').replace('V5 ',''))
        self.assertEqual(public[0]['content'],expected_system)
        self.assertEqual(public[1]['content'].split('\n\n',1)[0],
            prompt.USER_PREFIX.replace('v5_event_summary','event_summary').replace('original Continuous V5 ',''))
        a=prompt.batches(req,50,'original');b=prompt.batches(req,50,'public')
        self.assertEqual([x['expected'] for x in a],[x['expected'] for x in b])
        self.assertEqual([len(json.loads(x['messages'][1]['content'].split('\n\n',1)[1])) for x in a],
                         [len(json.loads(x['messages'][1]['content'].split('\n\n',1)[1])) for x in b])
        exported=Path(prompt.__file__).with_name('prompt_public.txt').read_text().strip()
        self.assertEqual(exported,prompt.PUBLIC_SYSTEM_PROMPT)

    def test_styles_do_not_share_generated_caches_or_priors(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=dict(setup_data(tmp),samples=1,prior_source='generated')
            public=dict(original,prompt_style='public')
            fake=FakeProvider()
            with patch.object(generation,'provider_client',return_value=fake):
                generation.run(original,True)
                old_path=cfg.prior_path(original,'B','OOD-4');old=old_path.read_bytes()
                generation.run(public,True)
                self.assertEqual(fake.count,2)
                self.assertNotEqual(old_path,cfg.prior_path(public,'B','OOD-4'))
                self.assertNotEqual(cfg.metric_root(original),cfg.metric_root(public))
                self.assertEqual(old_path.read_bytes(),old)
                generation.run(public,True);self.assertEqual(fake.count,2)
                # Identical fake responses give identical numerical priors: decoder is shared.
                pa=cfg.read_json(old_path);pb=cfg.read_json(cfg.prior_path(public,'B','OOD-4'))
                self.assertEqual(pa['zone_gate_records'],pb['zone_gate_records'])
                existing=dict(original,prior_source='existing')
                self.assertEqual(cfg.prior_path(existing,'B','OOD-4'),cfg.prior_path(dict(existing,prompt_style='public'),'B','OOD-4'))

    def test_original_corrected_array_and_outside_event(self):
        application.COLUMN_MODE='legacy'
        frame=pd.DataFrame({'T_pred__Z':REFERENCE['raw']})
        actual,_=application.apply_episode_gates(frame,REFERENCE['control'],REFERENCE['prior'])
        np.testing.assert_array_equal(actual.T_pred__Z.to_numpy(),REFERENCE['expected_corrected'])
        mask=application.episode_gate_profile(REFERENCE['prior']['zone_gate_records']['Z'],REFERENCE['control']['control_records']['Z'],95)==0
        np.testing.assert_array_equal(actual.T_pred__Z.to_numpy()[mask],frame.T_pred__Z.to_numpy()[mask])
        record=REFERENCE['control']['control_records']['Z']
        profile=np.arange(96)
        np.testing.assert_array_equal(application.aligned_daily_profile(profile,9120),np.tile(profile[1:],96))

    def test_mean_before_clip(self):
        req=prompt.build_request(REFERENCE['control'],REFERENCE['evidence'],'B','OOD-4')
        def vote(g,a):return {'Z':{1:{'event_gate':g,'confidence':'medium','reason':'fixture','phases':{p:{'phase_adjustment':a,'reason':'fixture'} for p in (1,2)}}}}
        prior=prompt.aggregate(req,[vote(.9,.4),vote(.1,-.2)])
        self.assertAlmostEqual(prior['zone_gate_records']['Z']['episode_gate_records'][0]['correction_gate'],.6)
        # mean(clip(.9+.4), clip(.1-.2)) would incorrectly give .5

    def test_batching_preserves_per_event_expectations(self):
        req=prompt.build_request(REFERENCE['control'],REFERENCE['evidence'],'B','OOD-4')
        cards=[];expected={}
        for i in range(51):
            c=dict(req['cards'][0],zone='Z%d'%i);cards.append(c);expected[c['zone']]=req['expected']['Z']
        request=dict(req,cards=cards,expected=expected)
        batches=prompt.batches(request,50)
        self.assertEqual([len(b['expected']) for b in batches],[50,1])

    def test_sample_counts_cache_and_decrease(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=setup_data(tmp);fake=FakeProvider()
            with patch.object(generation,'provider_client',return_value=fake):
                generation.run(s,False);self.assertEqual(fake.count,0)
                generation.run(s,True);self.assertEqual(fake.count,3)
                p3=cfg.prior_path(dict(s,prior_source='generated'),'B','OOD-4');saved3=p3.read_bytes()
                ten=dict(s,samples=10);generation.run(ten,True);self.assertEqual(fake.count,10)
                self.assertEqual(cfg.read_json(cfg.prior_path(dict(ten,prior_source='generated'),'B','OOD-4'))['metadata']['n_runs'],10)
                generation.run(s,True);self.assertEqual(fake.count,10);self.assertEqual(p3.read_bytes(),saved3)
                with self.assertRaises(ValueError):generation.preview(dict(s,model='changed-model'))

    def test_retries_are_not_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=dict(setup_data(tmp),samples=1,max_attempts=2);fake=FakeProvider(fail_first=1)
            with patch.object(generation,'provider_client',return_value=fake),patch.object(generation.time,'sleep'):
                generation.run(s,True)
            self.assertEqual(fake.count,2)
            prior=cfg.read_json(cfg.prior_path(dict(s,prior_source='generated'),'B','OOD-4'))
            self.assertEqual(prior['metadata']['n_runs'],1)

    def test_offline_application_partial_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=setup_data(tmp)
            out=application.run(s)
            scores=pd.read_csv(out/'case_metrics.csv')
            self.assertEqual(len(scores),23)
            self.assertEqual(scores.Points.nunique(),1)
            self.assertTrue(pd.read_csv(out/'failures.csv').empty)
            self.assertEqual(cfg.read_json(out/'status.json')['completed_selected_cases'],1)
            with patch.object(application,'process_case',side_effect=AssertionError('should reuse cache')):
                application.run(s)
            result=visualization.report(s,make_figures=False)
            self.assertIsNotNone(result)
            prior=cfg.prior_path(s,'B','OOD-4');prior.rename(prior.with_suffix('.retained'))
            application.run(s)
            self.assertTrue(pd.read_csv(out/'case_metrics.csv').empty)
            self.assertEqual(len(pd.read_csv(out/'failures.csv')),1)
            self.assertFalse(cfg.read_json(out/'status.json')['is_complete'])

    def test_original_metric_reference_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=setup_data(tmp);out=application.run(s)
            scores=pd.read_csv(out/'case_metrics.csv',dtype={k:str for k in cfg.KEYS})
            original=scores.loc[scores.Postprocess_Method.eq(cfg.OURS),cfg.KEYS+['MAE','RMSE','Points']]
            original.to_csv(s['existing_priors'].parent/'new_case_metrics.csv',index=False)
            visualization.report(s,make_figures=False)
            reference=pd.read_csv(out/'figures/original_result_reproduction.csv')
            self.assertEqual(len(reference),1)
            self.assertAlmostEqual(reference.MAE_delta.abs().max(),0)
            self.assertTrue(reference.Same_Points.all())

    def test_exact_air_mode_is_separate(self):
        frame=pd.DataFrame({'AUX_GYM:Zone Mean Air Temperature [C](TimeStep)':[1],
                            'GYM:Zone Mean Air Temperature [C](TimeStep)':[2]})
        application.COLUMN_MODE='legacy';legacy=application.find_zone_column(frame,'GYM',False)
        application.COLUMN_MODE='exact_air';exact=application.find_zone_column(frame,'GYM',False)
        self.assertTrue(legacy.startswith('AUX_GYM:'));self.assertTrue(exact.startswith('GYM:'))
        application.COLUMN_MODE='legacy'

    def test_plot_partial_one_ood(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=setup_data(tmp);application.run(s)
            try:import matplotlib
            except ImportError:self.skipTest('matplotlib unavailable')
            matplotlib.use('Agg')
            with patch('matplotlib.pyplot.show'):
                visualization.report(s)
            self.assertTrue((cfg.metric_root(s)/'figures/training_days_MAE.png').is_file())


if __name__=='__main__':unittest.main()
