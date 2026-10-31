# Repository monitoring and Gmail alerts

Monitoring is owned by GitHub Actions and does not require Codex to run.
The chat-based monitoring automation was paused.

## What runs

- `Review Compass health monitor`: every hour at minute 23 UTC and manually from
  Actions. Tests the committed Flask app/model locally in the runner and checks
  hosted-site reachability with three attempts before failing.
- Existing `Validate notebook environment`: full-data analysis and app checks
  on code changes. Failed main-branch runs are included in email alerts.
- `Email failures to Gmail`: listens for failed main-branch checks. Successful
  and cancelled checks, pull-request branches and forks do not send mail.
  Each failed run can produce one alert; an ongoing outage can therefore produce
  a new email on each failed hourly run. It does not send recovery/success emails.

The hosted Site is private. A working ChatGPT sign-in page establishes only gate
reachability, not authenticated product inference. The Flask app is tested inside
Actions; there is no deployed Flask server to monitor. Actions schedules are
best-effort and can be delayed; GitHub can disable schedules in inactive public
repositories after 60 days. This is not an external guaranteed-uptime service.

## Direct Gmail delivery: one-time activation

Open repository Settings > Secrets and variables > Actions and add:

| Repository secret | Value |
| --- | --- |
| `ALERT_GMAIL_USER` | The Gmail address used to send alerts |
| `ALERT_GMAIL_APP_PASSWORD` | An app password for that sender account, not its normal password |
| `ALERT_EMAIL_TO` | Your destination Gmail address; it may be the sender address |

[Open the repository secret settings](https://github.com/Ashik-Mohammed-S/Sentiment_Based_Product_Recommendation_System/settings/secrets/actions).

Google requires 2-Step Verification for app passwords; account or Workspace policy
may make app passwords unavailable. See [Google's app-password instructions](https://support.google.com/accounts/answer/185833).
Add credentials directly in GitHub Secrets, never in source files, issues or chat.
The GitHub connector used for this change cannot configure these secrets.

After adding them, open Actions > Email failures to Gmail > Run workflow, select
main and enable `test_delivery`. This sends one clearly labeled test email.
Until the secrets are added, direct Gmail delivery is **not active**: the mail
workflow reports a configuration warning and sends nothing. A successful Actions
run without credentials is not proof that email delivery works.

Mail uses Python's standard library and verified TLS to smtp.gmail.com:465.
The secrets-bearing workflow checks out only trusted main source, has read-only
repository permissions and does not consume triggering-run artifacts. It never
prints credentials. SMTP acceptance does not guarantee inbox delivery; check spam.

## Alternative: GitHub's own emails, without Gmail credentials

Set your verified Gmail address as your GitHub notification email. In
[GitHub notification settings](https://github.com/settings/notifications), under
System > Actions, enable Email and Only notify for failed workflows, then Save.
Watch this repository as needed. These account settings cannot be enabled by a
repository commit. Scheduled-run notifications are sent to the schedule's actor;
re-enable the scheduled workflow under your own account if necessary.

See [GitHub's Actions email instructions](https://docs.github.com/en/subscriptions-and-notifications/how-tos/managing-github-actions-notifications)
and [scheduled-run notification behavior](https://docs.github.com/en/actions/concepts/workflows-and-actions/notifications-for-workflow-runs).

Use either direct SMTP or native GitHub emails to avoid duplicate messages.
To stop periodic checks, disable Review Compass health monitor in Actions.
