# Security Policy

If you believe you have found a security vulnerability, please do not open a public issue.

Instead, use your hosting provider’s private vulnerability reporting (e.g. GitHub Security Advisories), or contact the maintainers privately.

## Notes

`agent-streams` runs local shell commands (`git`, `tmux`, and the agent CLI such as `claude`). Treat prompts as code: do not run untrusted prompts.
