"""Gmail SMTP failure delivery using repository secrets. No third-party mail action."""
from email.message import EmailMessage
import json
import os
from pathlib import Path
import re
import smtplib
import ssl

FAILURES={'failure','timed_out','startup_failure','action_required'}

def build_message(event,env):
    test=env.get('GITHUB_EVENT_NAME')=='workflow_dispatch' and env.get('TEST_DELIVERY')=='true'
    run=event.get('workflow_run',{})
    repo=env.get('GITHUB_REPOSITORY','')
    if not test:
        if env.get('GITHUB_EVENT_NAME')!='workflow_run' or run.get('conclusion') not in FAILURES:return None
        if run.get('head_branch')!='main' or run.get('head_repository',{}).get('full_name')!=repo:return None
    message=EmailMessage()
    message['Subject']=('[TEST] ' if test else '[FAILURE] ')+'Review Compass repository alert'
    run_id=int(env.get('GITHUB_RUN_ID','0')) if test else int(run['id'])
    link=f'https://github.com/{repo}/actions/runs/{run_id}'
    if test:body='Test email: GitHub Actions can deliver alerts to this address.'
    else:body=f"Workflow: {run.get('name','Repository check')}\nResult: {run['conclusion']}\nCommit: {run.get('head_sha','unknown')}"
    message.set_content(body+'\n\nOpen the run for details:\n'+link+'\n\nHosted sign-in-gate checks do not prove authenticated app health. The Flask checks run inside GitHub Actions, not on a deployed Flask server.\n')
    return message

def deliver(event,env,connect=smtplib.SMTP_SSL):
    message=build_message(event,env)
    if message is None:
        print('No failed main-branch check to notify.');return False
    required=['ALERT_GMAIL_USER','ALERT_GMAIL_APP_PASSWORD','ALERT_EMAIL_TO']
    missing=[key for key in required if not env.get(key)]
    if missing:
        print('::warning::Direct Gmail delivery is not configured. Add the three alert secrets described in MONITORING.md. Native GitHub failure-email settings are a separate option.')
        if env.get('TEST_DELIVERY')=='true':raise ValueError('Cannot test email delivery before configuring alert secrets')
        return False
    sender=env['ALERT_GMAIL_USER'].strip();recipient=env['ALERT_EMAIL_TO'].strip()
    for address in [sender,recipient]:
        if not re.fullmatch(r'[^\s<>@,;]+@[^\s<>@,;]+\.[^\s<>@,;]+',address):raise ValueError('Configure a single valid alert email address')
    message['From']=sender;message['To']=recipient
    password=env['ALERT_GMAIL_APP_PASSWORD'].replace(' ','')
    with connect('smtp.gmail.com',465,context=ssl.create_default_context(),timeout=30) as smtp:
        smtp.login(sender,password)
        smtp.send_message(message)
    print('Failure notification accepted by Gmail SMTP.');return True

if __name__=='__main__':
    path=os.environ.get('GITHUB_EVENT_PATH')
    event=json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    try:deliver(event,os.environ)
    except Exception as error:
        print('::error::Email delivery failed ('+type(error).__name__+'). Check alert secrets and sender account settings; credentials were not logged.')
        raise SystemExit(1)
