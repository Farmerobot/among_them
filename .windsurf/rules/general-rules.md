---
trigger: always_on
---

# General rules
- Use Conventional Commits (no scopes) when generating commit messages:
  - Never mention specific files or paths
  - Explain why, not what
  - Prefer no body unless 1 line is not enough to explain the commit
- Use Poetry for python dependencies
- Do not create any temporary comments or comments that do not make sense only in the context of the conversation. Every comment is permanent documentation
- Place imports at the top of files unless you have a good reason (e.g. lazy import)