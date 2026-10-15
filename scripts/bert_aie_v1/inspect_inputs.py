"""Read only existing permitted model/data metadata, never protected partitions."""
from collections import Counter
from common import CANONICAL, read, rows
base = CANONICAL / 'results/cnn_adoption_v1'
p = read(base / 'revision_r2/data_protocol.json')
print('protocol_keys', list(p))
membership = p['source_role_membership']
print('membership_type', type(membership).__name__)
print('membership_counts', {k:len(v) for k,v in membership.items()})
used = {x for values in membership.values() for x in values}
available = [r for r in rows(base / 'source_inventory.jsonl') if r['source_group_id'] not in used]
print('unused_sources', dict(Counter(r['genre'] for r in available)))
print('unused_groups', {g:len({r['source_group_id'] for r in available if r['genre']==g}) for g in {r['genre'] for r in available}})
print('canonical_model_policy', read(CANONICAL / 'configs/project.json')['models']['detector'])
print('prepared_roles', [p.name for p in (base / 'prepared/token_r2').iterdir() if p.is_dir()])
