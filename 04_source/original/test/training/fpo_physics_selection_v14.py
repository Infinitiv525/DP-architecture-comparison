"""Prospective physics-first selection while separately retaining best data fit."""
def checkpoint_key(reports):
    score=sum(r['score'] for r in reports)/len(reports)
    ratios=[r['interior_div_rms']/r['mass_limit'] if not r['failure'] else 1e6 for r in reports]
    passed=all(not r['failure'] and r['mass_pass'] for r in reports)
    return (0 if passed else 1,score if passed else max(ratios),score)
