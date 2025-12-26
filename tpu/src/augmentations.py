# src/augmentations.py
from __future__ import annotations
import random, copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

MAX_GRID_SIZE, MAX_SYMBOLS = 100, 10

# ---------- Minimal helpers (local; no external deps) ----------
def is_grid(g: Any) -> bool:
    try:
        if not isinstance(g, list) or not g: return False
        h, w = len(g), len(g[0])
        if not (0 < h <= MAX_GRID_SIZE and 0 < w <= MAX_GRID_SIZE): return False
        return all(isinstance(r, list) and len(r)==w and all(isinstance(x,int) and 0<=x<MAX_SYMBOLS for x in r) for r in g)
    except Exception:
        return False

def _dom(g: List[List[int]]) -> int:
    flat = [c for r in g for c in r]
    return max(set(flat), key=flat.count) if flat else 0

def _pad_to(g: List[List[int]], H: int, W: int) -> List[List[int]]:
    d = _dom(g)
    return [r + [d]*(W-len(r)) for r in g] + [[d]*W]*(H-len(g))

def _combine_h(grids: List[List[List[int]]]) -> Optional[List[List[int]]]:
    H = max(len(g) for g in grids); W = sum(len(g[0]) for g in grids)
    if H>MAX_GRID_SIZE or W>MAX_GRID_SIZE: return None
    outs = []
    for r in range(H):
        row=[]
        for g in grids:
            d = _dom(g); w=len(g[0])
            row.extend(g[r] if r<len(g) else [d]*w)
        outs.append(row)
    return outs

def _combine_v(grids: List[List[List[int]]]) -> Optional[List[List[int]]]:
    H = sum(len(g) for g in grids); W = max(len(g[0]) for g in grids)
    if H>MAX_GRID_SIZE or W>MAX_GRID_SIZE: return None
    outs=[]
    for g in grids:
        d=_dom(g)
        for r in g: outs.append(r + [d]*(W-len(r)))
    return outs

def _combine_with_sep(grids: List[List[List[int]]], direction: str, sep: int) -> Optional[List[List[int]]]:
    if direction=="horizontal":
        W = max(len(g[0]) for g in grids); rows=[]
        for i,g in enumerate(grids):
            if i>0: rows.append([sep]*W)
            d=_dom(g)
            rows += [r + [d]*(W-len(r)) for r in g]
        H=len(rows)
        if H>MAX_GRID_SIZE or W>MAX_GRID_SIZE: return None
        return rows
    # vertical
    H = max(len(g) for g in grids)
    base = grids[0]; d0=_dom(base); W0=len(base[0])
    rows=[(base[r][:] if r<len(base) else [d0]*W0) for r in range(H)]
    for g in grids[1:]:
        for r in range(H): rows[r].append(sep)
        d=_dom(g); W=len(g[0])
        for r in range(H): rows[r].extend(g[r] if r<len(g) else [d]*W)
    W=max(len(r) for r in rows)
    if H>MAX_GRID_SIZE or W>MAX_GRID_SIZE: return None
    return rows

def _combine_grid_rc(grids: List[List[List[int]]], R: int, C: int) -> Optional[List[List[int]]]:
    cells = list(grids)
    if len(cells) < R*C:
        filler = cells[-1]
        cells += [filler]*(R*C - len(cells))
    row_h=[0]*R; col_w=[0]*C
    for idx,g in enumerate(cells[:R*C]):
        rr,cc = divmod(idx,C)
        row_h[rr]=max(row_h[rr],len(g))
        col_w[cc]=max(col_w[cc],len(g[0]))
    H,W=sum(row_h),sum(col_w)
    if H>MAX_GRID_SIZE or W>MAX_GRID_SIZE: return None
    out=[[0]*W for _ in range(H)]
    r0=0; idx=0
    for rr in range(R):
        c0=0
        for cc in range(C):
            g=cells[idx]; idx+=1
            gh,gw=len(g),len(g[0]); ch, cw = row_h[rr], col_w[cc]; d=_dom(g)
            for r in range(ch):
                for c in range(cw):
                    out[r0+r][c0+c] = g[r][c] if (r<gh and c<gw) else d
            c0 += cw
        r0 += row_h[rr]
    return out

def combine(grids: List[List[List[int]]], method: str, sep_color: int) -> Optional[List[List[int]]]:
    if not grids: return None
    if method=='horizontal': return _combine_h(grids)
    if method=='vertical':   return _combine_v(grids)
    if method=='color_separator_horizontal': return _combine_with_sep(grids,'horizontal',sep_color)
    if method=='color_separator_vertical':   return _combine_with_sep(grids,'vertical',sep_color)
    if method.startswith('grid_'):
        try:
            r,c = method.split('_',1)[1].split('x')
            return _combine_grid_rc(grids,int(r),int(c))
        except Exception: return None
    return None

def shift_colors(g: List[List[int]], inc: int) -> List[List[int]]:
    if not inc: return g
    return [[(v+inc)%10 for v in row] for row in g]

# ---------- Base class ----------
class AugmentationBase:
    def __init__(self, name: str): self.name=name
    def augment(self, task: Dict, num_variants: int) -> Tuple[List[Dict], List[Dict]]: raise NotImplementedError
    def _ok(self, t: Dict) -> bool:
        try:
            return bool(t.get('train')) and bool(t.get('test')) and is_grid(t['test'][0]['input']) and is_grid(t['test'][0]['output']) and all(is_grid(x['input']) and is_grid(x['output']) for x in t['train'])
        except Exception: return False

# ---------- Mixup ----------
class MixupAugmentation(AugmentationBase):
    def __init__(self, overlay_mode="bernoulli", mix_ratio=0.5, color_shift_increment=1):
        super().__init__("mixup")
        self.overlay_mode=overlay_mode; self.mix_ratio=float(max(0,min(1,mix_ratio))); self.shift=color_shift_increment; self.pool=[]
    def set_task_pool(self, tasks: List[Dict]): self.pool=[t for t in tasks if self._ok(t)]
    def _pad(self,g,h,w): return _pad_to(g,h,w)
    def _mix(self,g1,g2):
        h=max(len(g1),len(g2)); w=max(len(g1[0]),len(g2[0]))
        if h==0 or w==0 or h>MAX_GRID_SIZE or w>MAX_GRID_SIZE: return None
        a=self._pad(g1,h,w); b=self._pad(g2,h,w)
        if self.overlay_mode=="add_mod":   return [[(x+y)%10 for x,y in zip(r1,r2)] for r1,r2 in zip(a,b)]
        if self.overlay_mode=="replace_nonzero": return [[(y if y!=0 else x) for x,y in zip(r1,r2)] for r1,r2 in zip(a,b)]
        rng=np.random.random((h,w)); out=[]
        for i in range(h):
            row=[]
            for j in range(w):
                fg, bg = b[i][j], a[i][j]
                row.append(fg if (rng[i,j]<self.mix_ratio and fg!=0) else bg)
            out.append(row)
        return out
    def _pairs(self,a,b,n):
        from itertools import cycle, islice
        A,B=(a,b) if len(a)>=len(b) else (b,a)
        return list(islice(zip(A,cycle(B)),n))
    def augment(self, task: Dict, num_variants: int):
        if not self.pool: return [],[]
        outs=[]
        for _ in range(num_variants):
            t2 = copy.deepcopy(random.choice(self.pool))
            n = max(len(task['train']), len(t2['train']))
            mixed_train=[]
            for ex1,ex2 in self._pairs(task['train'], t2['train'], n):
                mi=self._mix(ex1['input'], shift_colors(ex2['input'], self.shift))
                mo=self._mix(ex1['output'],shift_colors(ex2['output'],self.shift))
                if mi and mo: mixed_train.append({'input':mi,'output':mo})
            mti=self._mix(task['test'][0]['input'], shift_colors(t2['test'][0]['input'], self.shift))
            mto=self._mix(task['test'][0]['output'],shift_colors(t2['test'][0]['output'],self.shift))
            cand={'train':mixed_train,'test':[{'input':mti,'output':mto}]}
            if self._ok(cand): outs.append(cand)
        return outs,[]

# ---------- Combine helpers (used by both combine variants) ----------
_METHODS_DEFAULT = ['horizontal','vertical','color_separator_horizontal','color_separator_vertical','grid_2x1','grid_1x2','grid_2x2']

def _make_variant(t: Dict, shift: int) -> Dict:
    if not shift: return copy.deepcopy(t)
    u=copy.deepcopy(t)
    for ex in u['train']+u['test']:
        for k in ('input','output'):
            if k in ex and ex[k] is not None: ex[k]=shift_colors(ex[k], shift%10)
    return u

def _combine_task_list(tasks: List[Dict], method: str, sep_color: int, augment_train_pairs: bool) -> Optional[Dict]:
    if augment_train_pairs:
        L = max(len(x['train']) for x in tasks)
        train=[]
        for i in range(L):
            inp = [tasks[k]['train'][i % len(tasks[k]['train'])]['input']  for k in range(len(tasks))]
            out = [tasks[k]['train'][i % len(tasks[k]['train'])]['output'] for k in range(len(tasks))]
            ci = combine(inp, method, sep_color); co = combine(out, method, sep_color)
            if ci is None or co is None: return None
            train.append({'input':ci,'output':co})
    else:
        train = tasks[0]['train']
    test_in  = combine([t['test'][0]['input']  for t in tasks], method, sep_color)
    test_out = combine([t['test'][0]['output'] for t in tasks], method, sep_color)
    if test_in is None or test_out is None: return None
    return {'train':train,'test':[{'input':test_in,'output':test_out}]}

# ---------- Combine (single-task self-combine) ----------
class CombineAugmentation(AugmentationBase):
    def __init__(self, max_tasks_to_combine=3, augment_train_pairs=True, methods=None, separator_color=9, **_):
        super().__init__("combine")
        self.k=max(2,int(max_tasks_to_combine)); self.do_pairs=bool(augment_train_pairs)
        self.methods=(methods or _METHODS_DEFAULT); self.sep=int(separator_color)
    def augment(self, task: Dict, num_variants: int):
        outs=[]
        for _ in range(num_variants):
            m=random.choice(self.methods)
            k=random.randint(2,self.k)
            ts=[_make_variant(task,i) for i in range(k)]
            cand=_combine_task_list(ts,m,self.sep,self.do_pairs)
            if cand and AugmentationBase._ok(self,cand): outs.append(cand)
        return outs,[]

# ---------- Mixup+Combine (multi-task combine) ----------
class MixupCombineAugmentation(AugmentationBase):
    def __init__(self, max_tasks_to_combine=2, augment_train_pairs=True, methods=None, separator_color=9, **_):
        super().__init__("mixup_combine")
        self.k=max(2,int(max_tasks_to_combine)); self.do_pairs=bool(augment_train_pairs)
        self.methods=(methods or ['horizontal','vertical','color_separator_horizontal','color_separator_vertical'])
        self.sep=int(separator_color); self.pool=[]
    def set_task_pool(self, tasks: List[Dict]): self.pool=[t for t in tasks if AugmentationBase._ok(self,t)]
    def augment(self, task: Dict, num_variants: int):
        if not self.pool: return [],[]
        outs=[]
        for _ in range(num_variants):
            m=random.choice(self.methods); k=min(self.k, 1+len(self.pool))
            others=random.sample(self.pool, k-1) if k>1 else []
            ts=[_make_variant(t,i) for i,t in enumerate([task]+others)]
            cand=_combine_task_list(ts,m,self.sep,self.do_pairs)
            if cand and AugmentationBase._ok(self,cand): outs.append(cand)
        return outs,[]

# ---------- Simple augs ----------
class OrderAugmentation(AugmentationBase):
    def __init__(self): super().__init__("order")
    def augment(self, task: Dict, num_variants: int):
        outs=[]; 
        for _ in range(num_variants):
            t=copy.deepcopy(task); random.shuffle(t['train']); outs.append(t)
        return outs,[]

class InputOutputSwapAugmentation(AugmentationBase):
    def __init__(self, swap_probability=0.5):
        super().__init__("input_output_swap"); self.p=float(max(0,min(1,swap_probability)))
    def augment(self, task: Dict, num_variants: int):
        outs=[]
        for _ in range(num_variants):
            t=copy.deepcopy(task)
            for ex in t['train']:
                if random.random()<self.p: ex['input'],ex['output']=ex['output'],ex['input']
            outs.append(t)
        return outs,[]

class GeometricColorAugmentation(AugmentationBase):
    def __init__(self): 
        super().__init__("geometric_color")
        self.transforms=[lambda a:a, lambda a:np.rot90(a,1), lambda a:np.rot90(a,2), lambda a:np.rot90(a,3), lambda a:np.fliplr(a), lambda a:np.flipud(a)]
    def augment(self, task: Dict, num_variants: int):
        outs=[]
        for _ in range(num_variants):
            t=copy.deepcopy(task); T=random.choice(self.transforms); perm=list(range(10)); random.shuffle(perm)
            cmap={i:perm[i] for i in range(10)}
            for ex in t['train']+t['test']:
                for k in ('input','output'):
                    g=np.array([[cmap[v] for v in r] for r in ex[k]],dtype=int)
                    x=T(g); ex[k]=x.tolist() if isinstance(x,np.ndarray) else x
            outs.append(t)
        return outs,[]

# ---------- Manager (same public API) ----------
class AugmentationManager:
    def __init__(self, config):
        self.augs: Dict[str, Tuple[AugmentationBase,float]]={}
        if getattr(config,'geometric_color',{}).get('enabled'):
            self.augs['geometric_color']=(GeometricColorAugmentation(), config.geometric_color.get('weight',1.0))
        if getattr(config,'order',{}).get('enabled'):
            self.augs['order']=(OrderAugmentation(), config.order.get('weight',1.0))
        if getattr(config,'mixup',{}).get('enabled'):
            m=config.mixup
            self.augs['mixup']=(MixupAugmentation(m.get('overlay_mode','bernoulli'), float(m.get('mix_ratio',0.5)), int(m.get('color_shift_increment',1))), m.get('weight',0.5))
        if getattr(config,'input_output_swap',{}).get('enabled'):
            s=config.input_output_swap
            self.augs['input_output_swap']=(InputOutputSwapAugmentation(float(s.get('swap_probability',0.5))), s.get('weight',1.0))
        if getattr(config,'combine',{}).get('enabled'):
            c=config.combine
            self.augs['combine']=(CombineAugmentation(int(c.get('max_tasks_to_combine',2)), bool(c.get('augment_train_pairs',True)), c.get('methods'), int(c.get('separator_color',9))), c.get('weight',1.0))
        if getattr(config,'mixup_combine',{}).get('enabled'):
            c=config.mixup_combine
            self.augs['mixup_combine']=(MixupCombineAugmentation(int(c.get('max_tasks_to_combine',2)), bool(c.get('augment_train_pairs',True)), c.get('methods'), int(c.get('separator_color',9))), c.get('weight',1.0))
        self.names=[k for k,(a,w) in self.augs.items() if w>0]; self.weights=[w for k,(a,w) in self.augs.items() if w>0]
    def set_task_pool_for_mixup(self, tasks: List[Dict]):
        if 'mixup' in self.augs: self.augs['mixup'][0].set_task_pool(tasks)
        if 'mixup_combine' in self.augs: self.augs['mixup_combine'][0].set_task_pool(tasks)
    def generate_augmentations(self, task: Dict, num_augmentations: int, max_attempts_per_variant: int=6) -> List[Tuple[Dict,str]]:
        if not self.names: return []
        out=[]
        for _ in range(num_augmentations):
            for _try in range(max_attempts_per_variant):
                name = random.choices(self.names, weights=self.weights, k=1)[0]
                inst,_w = self.augs[name]
                ts,_ = inst.augment(task,1)
                if ts:
                    cand = ts[0]
                    if inst._ok(cand): out.append((cand,name)); break
        return out
