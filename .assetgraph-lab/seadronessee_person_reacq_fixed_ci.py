import seadronessee_stress_ci as S
_orig=S.choose_stress_window
def _choose(images,anns):
    r=_orig(images,anns)
    return r[0] if isinstance(r,tuple) else r
S.choose_stress_window=_choose
import seadronessee_person_reacq_ci as R
if __name__=='__main__': R.main()
