# Security Policy

## Reporting a Vulnerability

**Do not open a public issue for security problems.**

Use GitHub's private vulnerability reporting instead: go to the
[Security tab](https://github.com/santanu20/pdf-ocr-skill/security) and click
**Report a vulnerability**. Reports go only to the maintainer.

Please include, as far as you can:

- A minimal reproduction (command line, input PDF/page, expected vs. actual)
- The affected file or code path
- Impact assessment (crash, data loss, wrong output, code execution)

You should get an acknowledgement within a few days. This is a
single-maintainer project, so fix timelines depend on severity and available
time; I will keep you informed of progress.

## Scope

- Vulnerabilities in this repository's code: command-line handling, path
  handling, subprocess/venv bootstrap, model download and cache logic, and
  anything that lets a crafted PDF escape the local-only processing model.
- Concerns about third-party dependencies or upstream models should be
  reported to those projects; I will track and pin affected versions here
  once confirmed.

## Design guarantees worth probing

The tool is local-only by design: no document content, extracted text, or
page images is sent to any network service. The only network use is
first-run downloads of models and packages. Any path where document bytes
leave the machine is a reportable vulnerability.

## Supported Versions

Only the latest commit on `main` receives security fixes.
