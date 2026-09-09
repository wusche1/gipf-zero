"""Batched PUCT. Values are backed up via absolute colour, never blind negation."""
import math,time
import numpy as np
import torch
import gipf_engine as ge
from .model import encode

class Node:
    __slots__=('state','actions','p','n','w','children','actor')
    def __init__(self,state):
        self.state=state;self.actor=state.current_player;self.actions=None
        self.p=self.n=self.w=None;self.children={}
    def expand(self,logits):
        self.actions=np.asarray(self.state.legal_actions(),np.int32)
        if not len(self.actions):raise RuntimeError('Nonterminal state has no legal actions')
        v=logits[self.actions].astype(np.float64);v-=v.max();p=np.exp(v)
        self.p=p/p.sum();self.n=np.zeros(len(p),np.int32);self.w=np.zeros(len(p),np.float64)
    def select(self,cpuct):
        if hasattr(ge,'puct_select'):
            i=int(ge.puct_select(self.p,self.n,self.w,cpuct))
        else:
            q=np.divide(self.w,self.n,out=np.zeros_like(self.w),where=self.n>0)
            score=q+cpuct*self.p*math.sqrt(1+int(self.n.sum()))/(1+self.n)
            i=int(np.argmax(score))
        child=self.children.get(i)
        if child is None:
            s=self.state.clone();s.apply(int(self.actions[i]));child=Node(s);self.children[i]=child
        return i,child
    def policy(self):
        if self.n.sum()==0:return self.p.copy()
        return self.n/self.n.sum()

class BatchedMCTS:
    def __init__(self,model,device='cuda',cpuct=1.5,seed=0):
        self.model=model;self.device=device;self.cpuct=cpuct;self.rng=np.random.default_rng(seed)
        self.evaluations=0
    @torch.inference_mode()
    def evaluate(self,nodes):
        states=[n.state for n in nodes]
        encoded=ge.encode_batch(states) if hasattr(ge,'encode_batch') else np.stack([encode(state) for state in states])
        x=torch.from_numpy(encoded).to(self.device)
        logits,values=self.model(x)
        logits=logits.float().cpu().numpy();values=values.float().cpu().numpy()
        self.evaluations+=len(nodes)
        for n,p in zip(nodes,logits):n.expand(p)
        return values
    def search(self,roots,simulations=64,noise=False,deadline=None):
        if any(r.state.winner for r in roots):
            raise ValueError('cannot search from a terminal root')
        missing=[r for r in roots if r.actions is None and not r.state.winner]
        if missing:self.evaluate(missing)
        if noise:
            for r in roots:
                if r.actions is not None:
                    eta=self.rng.dirichlet(np.full(len(r.actions),10/len(r.actions)))
                    r.p=.75*r.p+.25*eta
        for simulation in range(simulations):
            if deadline is not None and time.monotonic()>=deadline:break
            pending=[];paths=[]
            for root in roots:
                node=root;path=[]
                while node.actions is not None and not node.state.winner:
                    i,child=node.select(self.cpuct);path.append((node,i));node=child
                if node.state.winner:
                    self.backup(path,float(node.state.winner))
                else:pending.append(node);paths.append(path)
            if pending:
                values=self.evaluate(pending)
                for node,path,value in zip(pending,paths,values):self.backup(path,float(value)*node.actor)
        return [r.policy() for r in roots]
    @staticmethod
    def backup(path,white_value):
        for parent,i in path:
            parent.n[i]+=1;parent.w[i]+=white_value*parent.actor

def choose_action(model,state,device='cpu',simulations=256,budget_ms=1000):
    if state.winner:
        raise ValueError('cannot choose an action from a terminal state')
    root=Node(state.clone());search=BatchedMCTS(model,device)
    pi=search.search([root],simulations,deadline=time.monotonic()+budget_ms/1000)[0]
    return int(root.actions[int(np.argmax(pi))]),{'simulations':int(root.n.sum()),'value':float(root.w.sum()/max(1,root.n.sum()))}
