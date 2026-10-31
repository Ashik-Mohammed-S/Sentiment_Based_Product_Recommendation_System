import unittest
from unittest.mock import MagicMock
from scripts.monitor_health import check_site,probe_site,check_local_app
from scripts.send_failure_email import deliver,build_message

class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.env={'GITHUB_EVENT_NAME':'workflow_run','GITHUB_REPOSITORY':'owner/repo'}
        self.event={'workflow_run':{'id':123,'name':'Health monitor','conclusion':'failure','head_branch':'main','head_repository':{'full_name':'owner/repo'},'head_sha':'abc'}}
    def test_only_failed_trusted_main_runs_are_notified(self):
        self.assertIsNotNone(build_message(self.event,self.env))
        for key,value in [('conclusion','success'),('conclusion','cancelled'),('head_branch','feature'),('head_repository',{'full_name':'fork/repo'})]:
            event={'workflow_run':{**self.event['workflow_run'],key:value}}
            self.assertIsNone(build_message(event,self.env))
    def test_missing_secrets_never_connect(self):
        connect=MagicMock();self.assertFalse(deliver(self.event,self.env,connect));connect.assert_not_called()
    def test_delivery_uses_tls_and_run_link(self):
        connect=MagicMock();env={**self.env,'ALERT_GMAIL_USER':'sender@gmail.com','ALERT_GMAIL_APP_PASSWORD':'test-only-placeholder','ALERT_EMAIL_TO':'recipient@gmail.com'}
        self.assertTrue(deliver(self.event,env,connect))
        smtp=connect.return_value.__enter__.return_value
        message=smtp.send_message.call_args.args[0]
        self.assertIn('https://github.com/owner/repo/actions/runs/123',message.get_content())
        self.assertNotIn(env['ALERT_GMAIL_APP_PASSWORD'],message.as_string())
        self.assertIsNotNone(connect.call_args.kwargs['context'])
    def test_transient_site_errors_retry(self):
        probe=MagicMock(side_effect=[OSError(), 'reachable']);sleep=MagicMock()
        self.assertEqual(check_site(probe=probe,sleep=sleep),'reachable');self.assertEqual(probe.call_count,2)
        with self.assertRaises(RuntimeError):check_site(probe=MagicMock(side_effect=OSError()),sleep=sleep)
    def test_private_gate_is_not_misreported_as_full_health(self):
        response=MagicMock();response.__enter__.return_value=response;response.status=200
        response.read.return_value=b'<title>Sign in required</title>Log in to access'
        self.assertIn('authenticated hosted inference is not checked',probe_site(MagicMock(return_value=response)))
        response.read.return_value=b'<h1>Unexpected error</h1>'
        with self.assertRaises(RuntimeError):probe_site(MagicMock(return_value=response))
    def test_local_model_is_operational(self):
        self.assertIn('passed',check_local_app())

if __name__=='__main__':unittest.main()
