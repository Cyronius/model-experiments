import concurrent.futures,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import campaign_budget as budget

class BudgetTests(unittest.TestCase):
 def test_concurrent_reservations_cannot_exceed_cap(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ledger=root/'ledger.json';ledger.write_text(json.dumps({'authorized_total_usd':10.,'jobs':[]}))
   def reserve(name,cap):
    try:budget.reserve({'name':name,'reserved_cap_usd':cap,'state_path':str(root/(name+'.json'))});return True
    except RuntimeError:return False
   with patch.object(budget,'LEDGER',ledger):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
     results=[pool.submit(reserve,'a',6.),pool.submit(reserve,'b',5.)];self.assertEqual(sum(f.result() for f in results),1)
    self.assertLessEqual(budget.sync()['committed_upper_bound_usd'],10.)
 def test_durable_reservation_survives_stale_controller_publish(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ledger=root/'ledger.json';initial={'authorized_total_usd':10.,'jobs':[]};ledger.write_text(json.dumps(initial))
   with patch.object(budget,'LEDGER',ledger):
    budget.reserve({'name':'a','reserved_cap_usd':3.,'state_path':str(root/'a.json')});ledger.write_text(json.dumps(initial));self.assertEqual(budget.sync()['committed_upper_bound_usd'],3.)
 def test_active_reservations_then_realized_cost(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);state=root/'pod.json';ledger=root/'ledger.json'
   state.write_text(json.dumps({'id':'owned-pod','deleted':False}))
   ledger.write_text(json.dumps({'authorized_total_usd':10.,'jobs':[{'name':'probe','reserved_cap_usd':2.,'state_path':str(state)}]}))
   with patch.object(budget,'LEDGER',ledger):
    self.assertEqual(budget.sync()['committed_upper_bound_usd'],2.)
    state.write_text(json.dumps({'id':'owned-pod','deleted':True,'estimated_total_usd':.91}))
    self.assertEqual(budget.sync()['committed_upper_bound_usd'],.91)
    with self.assertRaisesRegex(RuntimeError,'already been used'):budget.guard('probe',{'proposed_total_cap_usd':2.},{'campaign_cap_usd':10.})
 def test_guard_rejects_aggregate_overspend(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ledger=root/'ledger.json'
   ledger.write_text(json.dumps({'authorized_total_usd':10.,'jobs':[{'name':'probe','reserved_cap_usd':6.,'state_path':str(root/'a')},{'name':'other','reserved_cap_usd':5.,'state_path':str(root/'b')}]}))
   with patch.object(budget,'LEDGER',ledger):
    with self.assertRaisesRegex(RuntimeError,'exceed approved'):budget.guard('probe',{'proposed_total_cap_usd':6.},{'campaign_cap_usd':10.})
if __name__=='__main__':unittest.main()
