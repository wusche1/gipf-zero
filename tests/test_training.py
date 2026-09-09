import random
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
import gipf_engine as ge
from training.model import PolicyValue,ModelConfig,encode,SYMMETRIES,ROWS,COLS
from training.search import Node,BatchedMCTS
from training import evaluate as training_evaluate

def transformed(s,perm):
    d=s.serialize();b=d['board'];new=[0]*37
    for i,j in enumerate(perm):new[int(j)]=b[i]
    d['board']=new;return ge.State.from_dict(d)

def test_symmetry_commutes_with_rules():
    rng=random.Random(47);s=ge.State()
    for _ in range(160):
        if s.winner:s=ge.State()
        legal=s.legal_actions();a=rng.choice(legal)
        for perm,ap in SYMMETRIES:
            t=transformed(s,perm)
            assert set(t.legal_actions())==set(int(ap[x]) for x in legal)
            t.apply(int(ap[a]));u=s.clone();u.apply(a)
            assert t.serialize()==transformed(u,perm).serialize()
        s.apply(a)

def test_backup_same_actor_and_opponent():
    class Parent:
        def __init__(self,actor):self.actor=actor;self.n=np.zeros(1);self.w=np.zeros(1)
    p=[Parent(1),Parent(1),Parent(-1)]
    BatchedMCTS.backup([(x,0) for x in p],1)
    assert [x.w[0] for x in p]==[1,1,-1]

def test_tiny_batch_overfits():
    torch.set_num_threads(2);torch.manual_seed(5)
    model=PolicyValue(ModelConfig('mlp',64,1))
    state=ge.State();x=torch.tensor(np.stack([encode(state)]*8))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    history=[]
    for _ in range(35):
        p,v=model(x);loss=torch.nn.functional.cross_entropy(p[:,:42],torch.zeros(8,dtype=torch.long))+(v-1).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step();history.append(loss.item())
    assert np.isfinite(history).all()
    assert history[-1]<history[0]*.1

def test_search_legal_normalized_and_checkpoint(tmp_path):
    from training.model import load_model
    torch.set_num_threads(2);model=PolicyValue(ModelConfig('mlp',64,1)).eval()
    roots=[Node(ge.State()) for _ in range(4)]
    search=BatchedMCTS(model,'cpu');pi=search.search(roots,8)
    for root,p in zip(roots,pi):
        assert np.isclose(p.sum(),1);assert root.n.sum()==8
        assert int(root.actions[np.argmax(p)]) in root.state.legal_actions()
    path=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'config':{'kind':'mlp','width':64,'blocks':1}},path)
    loaded,_=load_model(path)
    x=torch.tensor(np.stack([encode(ge.State())]))
    assert torch.equal(model(x)[0],loaded(x)[0])

def test_search_rejects_terminal_roots():
    class Terminal:
        winner=1
        current_player=1
        def clone(self): return self
    model=PolicyValue(ModelConfig('mlp',16,1)).eval()
    root=Node(Terminal())
    with pytest.raises(ValueError,match='terminal root'):
        BatchedMCTS(model,'cpu').search([root],1)
    with pytest.raises(ValueError,match='terminal state'):
        from training.search import choose_action
        choose_action(model,Terminal(),'cpu',1,50)

def test_native_encode_batch_matches_python_encoder_on_legal_states():
    rng=random.Random(9182)
    states=[];state=ge.State()
    for _ in range(320):
        states.append(state.clone())
        if state.winner:
            state=ge.State()
        else:
            state.apply(rng.choice(state.legal_actions()))
    native=ge.encode_batch(states)
    python=np.stack([encode(state) for state in states])
    assert native.dtype==np.float32
    assert native.shape==(len(states),9,7,7)
    np.testing.assert_array_equal(native,python)


def test_evaluation_settles_an_opening_capture_and_delays_insertion_cutoff():
    class Opening:
        def __init__(self):
            self.ply=3;self.phase='push';self.winner=0;self.applied=[]
        def legal_actions(self):return [0]
        def apply(self,action):
            self.applied.append(action)
            if self.phase=='push':self.ply=4;self.phase='capture'
            else:self.phase='push'
    state=Opening()
    training_evaluate.settle_opening(state,4,random.Random(1))
    assert state.phase=='push' and len(state.applied)==2
    state.phase='capture'
    assert not training_evaluate.is_insertion_cutoff(state,4)
    state.phase='push'
    assert training_evaluate.is_insertion_cutoff(state,4)


def test_evaluation_resume_records_are_keyed_and_reject_mismatch(tmp_path):
    args=SimpleNamespace(opponent='random',simulations=8,move_seconds=.05,max_ply=240,seed=7,opening_plies=4,neural_budget_ms=0,threads=2,device='cpu',batch=1,baseline_simulations=None,baseline_depth=None)
    key=training_evaluate.evaluation_key(args,'checkpoint-digest')
    path=tmp_path/'evaluation.jsonl'
    path.write_text(json.dumps({'game':0,'evaluation_key':key})+'\n')
    records,completed=training_evaluate.load_resume_records(path,key,2)
    assert records[0]['game']==0 and completed=={0}
    with pytest.raises(ValueError,match='mismatch'):
        training_evaluate.load_resume_records(path,'other-digest',2)
    changed=SimpleNamespace(**vars(args));changed.batch=2
    assert training_evaluate.evaluation_key(changed,'checkpoint-digest')!=key


def test_evaluation_measurements_report_empty_and_populated_samples():
    assert training_evaluate.measurement_stats([])=={'count':0,'mean':None,'min':None,'max':None}
    assert training_evaluate.measurement_stats([2,4,6])=={'count':3,'mean':4.0,'min':2,'max':6}


def test_evaluation_outcomes_keep_colours_pairs_and_cutoffs_separate():
    records=[
        {'game':0,'neural_color':1,'outcome':'win','cutoff_reason':None},
        {'game':1,'neural_color':-1,'outcome':'loss','cutoff_reason':None},
        {'game':2,'neural_color':1,'outcome':'win','cutoff_reason':None},
        {'game':3,'neural_color':-1,'outcome':'win','cutoff_reason':None},
        {'game':4,'neural_color':1,'outcome':'cutoff','cutoff_reason':'insertion'},
        {'game':5,'neural_color':-1,'outcome':'loss','cutoff_reason':None},
        {'game':6,'neural_color':1,'outcome':'win','cutoff_reason':None},
    ]
    summary=training_evaluate.outcome_summary(records)
    assert summary['wins']==4 and summary['losses']==2 and summary['cutoffs']==1
    assert summary['insertion_cutoffs']==1 and summary['decision_cutoffs']==0
    pairs=training_evaluate.paired_opening_summary(records)
    assert pairs['complete_pairs']==3
    assert pairs['neural_sweeps']==1 and pairs['splits']==1 and pairs['pairs_with_cutoff']==1

def test_factorized_capture_head_shares_mask_decisions():
    from training.model import ACTIONS
    model=PolicyValue(ModelConfig('mlp',64,1,'factorized'))
    with torch.no_grad():
        model.policy.weight.zero_();model.policy.bias.zero_()
        # A unit score for taking the first cell on line0 must affect exactly
        # masks with that bit, independently of the other mask bits.
        cell=ge.geometry()['lines'][0][0];model.policy.bias[63+cell]=1
    p,_=model(torch.tensor(encode(ge.State())[None]))
    assert p.shape==(1,ACTIONS)
    assert p[0,42].item()==0
    assert p[0,43].item()==1
    assert p[0,45].item()==1
