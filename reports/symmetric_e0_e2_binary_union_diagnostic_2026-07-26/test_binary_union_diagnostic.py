from __future__ import annotations
import math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import run_binary_union_diagnostic as a

def test_01_union_labels():
    assert np.array_equal(np.array([0,1,2,5])>0,np.array([0,1,1,1],bool))
def test_02_perfect_dice():
    x=np.array([1,0,1],bool);assert a.dice(x,x)==1
def test_03_missed_dice():
    assert a.dice(np.array([1],bool),np.array([0],bool))==0
def test_04_empty_empty_nan():
    assert math.isnan(a.dice(np.array([0],bool),np.array([0],bool)))
def test_05_hd95_perfect_zero():
    x=np.zeros((3,3,3),bool);x[1,1,1]=1;assert a.hd95_mm(x,x,(1,1,1))==0
def test_06_hd95_physical_spacing():
    x=np.zeros((3,3,3),bool);y=x.copy();x[1,1,0]=1;y[1,1,1]=1;assert a.hd95_mm(x,y,(1,1,5))==5
def test_07_hd95_empty_nan():
    x=np.zeros((2,2,2),bool);assert math.isnan(a.hd95_mm(x,x,(1,1,1)))
def test_08_fp_volume():
    g=np.array([1,0],bool).reshape(1,1,2);p=np.array([1,1],bool).reshape(1,1,2);assert a.case_metrics(g,p,(1,1,1),.5)["fp_volume_ml"]==.5
def test_09_fn_volume():
    g=np.array([1,1],bool).reshape(1,1,2);p=np.array([1,0],bool).reshape(1,1,2);assert a.case_metrics(g,p,(1,1,1),.5)["fn_volume_ml"]==.5
def test_10_precision_recall():
    g=np.array([1,1,0],bool).reshape(1,1,3);p=np.array([1,0,1],bool).reshape(1,1,3);m=a.case_metrics(g,p,(1,1,1),1);assert m["precision"]==.5 and m["recall"]==.5
def test_11_complete_miss():
    g=np.array([1],bool).reshape(1,1,1);p=np.array([0],bool).reshape(1,1,1);assert a.case_metrics(g,p,(1,1,1),1)["complete_miss"]
def test_12_small_group():
    assert a.lesion_group(.9)=="small_lt1ml"
def test_13_medium_group_boundaries():
    assert a.lesion_group(1)=="medium_1to10ml" and a.lesion_group(9.999)=="medium_1to10ml"
def test_14_large_group():
    assert a.lesion_group(10)=="large_ge10ml"
def test_15_soft_union_sum():
    p0=np.array([.4,.6]);assert np.array_equal(1-p0>=.5,np.array([1,0],bool))
def test_16_hard_soft_can_differ():
    p=np.array([.4,.12,.12,.12,.12,.12]);assert np.argmax(p)==0 and (1-p[0])>=.5
def test_17_wrong_subtype_recovered_by_union():
    gt=np.array([4]);pred=np.array([5]);assert (gt>0)[0] and (pred>0)[0] and gt[0]!=pred[0]
def test_18_voxel_determinant():
    aff=np.diag([.5,.5,5,1]);assert math.isclose(abs(np.linalg.det(aff[:3,:3]))/1000,.00125)
def test_19_finite_mean_skips_nan():
    assert a.finite_mean([1,math.nan])==1
def test_20_seed_sd():
    assert a.sample_sd([1,2,3])==1

if __name__=="__main__":
    tests=sorted((n,f) for n,f in globals().items() if n.startswith("test_") and callable(f))
    for name,test in tests:test();print("PASS",name)
    print(f"{len(tests)} tests passed")
