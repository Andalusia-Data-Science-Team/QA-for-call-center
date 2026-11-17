import argparse
import asyncio
from collections import Counter
from copy import copy
from datetime import datetime, time
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from zipfile import ZipFile

from openpyxl import Workbook, load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('excel_reevaluation', Path(__file__).with_name('reevaluate_excel.py'))
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class ExcelReevaluationTests(unittest.TestCase):
    def fixture(self, folder):
        path = Path(folder) / 'reviews.xlsx'
        book = Workbook(); sheet = book.active; sheet.title = 'AI Quality Evaluation'
        sheet.append([])
        sheet.append(['', 'Patient Number', 'Date', 'Time', 'Agent Name', 'C2C', 'C2B', 'C2COM', 'NC', 'QA Evaluation', 'AI Evaluation', 'QA Validation', 'QA Name', 'AI Evaluation Accuracy', 'Score'])
        sheet.append(['', 533585356, datetime(2026,11,9), time(7), 'Agent', 0,1,0,0,'Human review','Old AI','Original validation','Reviewer','Failed','=IF(N3="Passed",1,0)'])
        sheet['B119']='=300*0.6'
        book.create_sheet('Score')['A1']='Preserve this sheet'
        book.save(path);book.close()
        manifest={'workbook':str(path),'sheet':sheet.title,'run_id':'test','start_col':16,'rows':{},'excel_schema_width':61}
        return path,manifest

    def test_phone_normalization(self):
        for value in [533585356,533585356.0,'٠٥٣٣٥٨٥٣٥٦','+966 53-358-5356','00966533585356']:
            self.assertEqual(runner.normalize_phone(value),'533585356')
        for value in [None,'=300*0.6','Total','123','15533585356']:
            with self.assertRaises(ValueError):runner.normalize_phone(value)

    def test_review_selection_skips_footer(self):
        with tempfile.TemporaryDirectory() as folder:
            path,d=self.fixture(folder)
            reviews=runner.read_reviews(path,d['sheet'])
            self.assertEqual(len(reviews),1)
            self.assertEqual(reviews[0]['row'],3)

    def test_candidate_date_and_duplicate_flags(self):
        review={'date':'2026-11-09','agent':'Agent','time':time(7)}
        a={'Call_ID':'A','Call_Date':datetime(2026,11,9,7),'Agent_Name':'Agent'}
        b=dict(a,Call_ID='B',Call_Date=datetime(2026,11,8,7))
        self.assertEqual(runner.select_candidate(review,[a,a,b])['Call_ID'],'A')
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            runner.select_candidate(review,[a,dict(a,Call_ID='B')])

    def flag(self, number=1):
        from app.services.regulation_catalog import regulation_items
        item=next(x for x in regulation_items() if x['type']=='C2B' and x['comment']!='Other')
        return {**item,'description':f'Issue {number}','transcript_excerpt':f'Evidence {number}','severity':'critical'}

    def test_appended_output_matches_actual_export_cells_styles_and_groups(self):
        from app.services.qa_results_export import build_customer_care_workbook
        with tempfile.TemporaryDirectory() as folder:
            path,d=self.fixture(folder);reviews=runner.read_reviews(path,d['sheet'])
            d['rows']['3']={'status':'completed','result':{'call_id':'chat','agent_name':'Agent','overall_assessment':'needs_review','compliance_flags':[self.flag()], '_call_metadata':{'call_duration_seconds':120}},'completed_utc':'2026-11-16T14:00:00Z'}
            projected=runner.export_calls(d,reviews)
            expected=load_workbook(BytesIO(build_customer_care_workbook(projected)))
            runner.write_output_sheet(d,reviews,Path(folder)/'results.json')
            actual=load_workbook(path)
            a=actual[d['sheet']]; e=expected.active
            for row in (1,2,3):
                for col in range(1,62):
                    x,y=a.cell(row,col+15),e.cell(row,col)
                    self.assertEqual(x.value,y.value,(row,col))
                    self.assertEqual(copy(x.fill),copy(y.fill),(row,col))
                    self.assertEqual(copy(x.font),copy(y.font),(row,col))
                    self.assertEqual(copy(x.border),copy(y.border),(row,col))
                    self.assertEqual(copy(x.alignment),copy(y.alignment),(row,col))
                    self.assertEqual(x.number_format,y.number_format,(row,col))
            self.assertEqual(a['AI2'].value,'Error 1')
            self.assertEqual(a['AI3'].value,self.flag()['type'])
            self.assertEqual(a['AJ3'].value,self.flag()['attribute'])
            self.assertEqual(a['AK3'].value,self.flag()['comment'])
            self.assertEqual(a['P1'].value,'Month')
            self.assertEqual(a['BX3'].value,'chat')
            self.assertIn('AI2:AL2',{str(x) for x in a.merged_cells.ranges})
            actual.close();expected.close()

    def test_all_flags_and_hyperlinks_survive_with_stable_style_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path,d=self.fixture(folder);reviews=runner.read_reviews(path,d['sheet'])
            # Distinct descriptions ensure all six findings remain visible in All Escalations.
            d['rows']['3']={'status':'completed','result':{'call_id':'chat','agent_name':'Agent','overall_assessment':'needs_review', 'conversation_link':'https://example.test/chat', 'compliance_flags':[self.flag(i) for i in range(1,7)]}}
            runner.write_output_sheet(d,reviews,Path(folder)/'results.json')
            w=load_workbook(path); count=len(w._cell_styles);w.close()
            runner.write_output_sheet(d,reviews,Path(folder)/'results.json')
            w=load_workbook(path);s=w[d['sheet']]
            self.assertEqual(len(w._cell_styles),count)
            self.assertIn('Issue 6',s['BV3'].value)
            self.assertEqual(s['BW3'].hyperlink.target,'https://example.test/chat')
            w.close()

    def test_pending_and_error_rows_are_never_passes_or_zero_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path,d=self.fixture(folder);reviews=runner.read_reviews(path,d['sheet'])
            for record,label in [({},'Pending re-evaluation'),({'status':'error','error':'No matching conversation'},'No matching conversation')]:
                d['rows']={'3':record} if record else {}
                runner.write_output_sheet(d,reviews,Path(folder)/'results.json')
                w=load_workbook(path);s=w[d['sheet']]
                self.assertIsNone(s['AE3'].value)
                self.assertIsNone(s['BJ3'].value)  # Export Last Call Result (AU + 15).
                self.assertIsNone(s['BN3'].value)  # Export Agent Classification (AY + 15).
                self.assertIn(label,s['BM3'].value)  # Clear Comment (AX + 15).
                w.close()

    def test_preserves_original_cells_and_unrelated_zip_parts(self):
        with tempfile.TemporaryDirectory() as folder:
            path,d=self.fixture(folder); reviews=runner.read_reviews(path,d['sheet'])
            with ZipFile(path,'a') as z:z.writestr('custom-preserve.xml','<preserve/>')
            with ZipFile(path) as z:before={n:z.read(n) for n in z.namelist()}
            w=load_workbook(path);original=[[c.value for c in row] for row in w[d['sheet']].iter_rows(max_col=15)];w.close()
            runner.write_output_sheet(d,reviews,Path(folder)/'results.json')
            with ZipFile(path) as z:
                for name,data in before.items():
                    if name not in {'xl/styles.xml','xl/worksheets/sheet1.xml'}:
                        self.assertEqual(z.read(name),data,name)
            w=load_workbook(path)
            self.assertEqual([[c.value for c in row] for row in w[d['sheet']].iter_rows(max_col=15)],original)
            self.assertEqual(w['Score']['A1'].value,'Preserve this sheet');w.close()

    def test_fresh_rerun_preserves_old_checkpoint_and_replaces_its_block(self):
        with tempfile.TemporaryDirectory() as folder:
            path,d=self.fixture(folder);reviews=runner.read_reviews(path,d['sheet'])
            old=Path(folder)/'old.json';d['fingerprint']='test';d['rows']={'3':{'status':'completed'}}
            old.write_text(json.dumps(d))
            with patch.object(runner,'ROOT',Path(folder)):
                details,fresh=runner.fresh_rerun(old,d)
            self.assertEqual(json.loads(old.read_text())['rows'],d['rows'])
            self.assertEqual(fresh['rows'],{})
            self.assertEqual(fresh['start_col'],16)
            self.assertTrue(Path(fresh['backup']).exists())
            self.assertEqual(fresh['previous_run'],str(old))

    def test_resume_recovers_expanded_sheet_when_checkpoint_width_is_stale(self):
        for stale_width in (0,14):
            with self.subTest(stale_width=stale_width),tempfile.TemporaryDirectory() as folder:
                path,d=self.fixture(folder);reviews=runner.read_reviews(path,d['sheet'])
                runner.write_output_sheet(d,reviews,Path(folder)/'results.json')
                d['excel_schema_width']=stale_width
                runner.write_output_sheet(d,reviews,Path(folder)/'results.json')
                self.assertEqual(d['excel_schema_width'],61)

    def test_expansion_refuses_to_overwrite_unrelated_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            path,d=self.fixture(folder);reviews=runner.read_reviews(path,d['sheet'])
            d['excel_schema_width']=14
            w=load_workbook(path);w[d['sheet']]['AD3']='Unrelated notes';w.save(path);w.close()
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'unrelated'):
                runner.write_output_sheet(d,reviews,Path(folder)/'results.json')
            self.assertEqual(path.read_bytes(),before)

    def test_single_chat_suppresses_database_insert_and_retains_export_metadata(self):
        import app.agent.nodes as nodes
        call_id='00000000-0000-0000-0000-000000000001'
        call=SimpleNamespace(call_id=call_id,Patient_Phone='533585356',call_duration_seconds=120,
            call_date='2026-11-09',archive_datetime=None,agent_name='Agent',department='Care',channel='WhatsApp',conversation_link=None,business_unit='AHJ')
        result=SimpleNamespace(model_dump=lambda **kwargs:{'overall_assessment':'pass'})
        class Agent:
            async def analyze(self,call):
                nodes.insert_qa_result(result,call)
                return result
        modules={'CM_DB_Handler':SimpleNamespace(CMDatabaseHandler=MagicMock()),'Chat_Retriever':SimpleNamespace(retrieve_chats_by_agent_email=lambda *a,**k:SimpleNamespace(empty=False),df_to_batch=lambda df:SimpleNamespace(calls=[call]))}
        with patch.dict(sys.modules,modules),patch.object(nodes,'insert_qa_result',side_effect=AssertionError('real write')) as insert:
            output=asyncio.run(runner.evaluate_chat(Agent(),call_id,call.Patient_Phone,Path('unused')))
            insert.assert_not_called()
            self.assertEqual(output['_call_metadata']['call_duration_seconds'],120)

    def test_actual_workbook_has_38_reviews(self):
        self.assertEqual(len(runner.read_reviews(runner.DEFAULT_BOOK,runner.DEFAULT_SHEET)),38)


if __name__=='__main__':unittest.main()
