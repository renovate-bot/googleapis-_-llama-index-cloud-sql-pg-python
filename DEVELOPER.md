# DEVELOPER.md

## Versioning

This library follows [Semantic Versioning](http://semver.org/).

## Processes

### Conventional Commit messages

This repository uses tool [Release Please](https://github.com/googleapis/release-please) to create GitHub and PyPi releases. It does so by parsing your
git history, looking for [Conventional Commit messages](https://www.conventionalcommits.org/),
and creating release PRs.

Learn more by reading [How should I write my commits?](https://github.com/googleapis/release-please?tab=readme-ov-file#how-should-i-write-my-commits)

### Modifying existing classes
When modifying any existing classes, ensure the required libraries are installed in "editable mode" using the command `pip install -e .` from your project's root directory.

This allows your code to dynamically reflect any changes you make to the library, enabling efficient local testing of your new methods.

## Dependencies

The core dependencies needed for the working of this library are mentioned in `requirements.txt`, but it's version ranges along with the testing dependencies are set in `pyproject.toml`.

## Testing

### Formatting

This repo has a lint checker test which is run on every PR  as set up in `.github/workflows/lint.yml`.
You can run the formatter locally by following these steps.

1.  Download all test dependencies
    ```bash
    pip install .[test]
    ```

2.  Run the black code formatter
    ```bash
    black .
    ```

3.  Run isort to sort all imports
    ```bash
    isort .
    ```

4.  Run mypy type checker
    ```bash
    mypy .
    ```

### Run tests locally

1. Set environment variables for `INSTANCE_ID`, `DATABASE_ID`, `REGION`, `DB_USER`, `DB_PASSWORD`, `IAM_ACCOUNT`.

2. Run pytest to automatically run all tests:

    ```bash
    pytest
    ```

Notes:

* Tests use both IAM and built-in authentication.
  * Learn how to set up a built-in databases user at [Cloud SQL built-in database authentication](https://cloud.google.com/sql/docs/postgres/built-in-authentication).
  * Local tests will run against your `gcloud` credentials. Use `gcloud` to login with your personal account or a service account. This account will be used to run IAM tests. Learn how to set up access to the database at [Manage users with IAM database authentication](https://cloud.google.com/sql/docs/postgres/add-manage-iam-users). The "IAM_ACCOUNT" environment variable is also used to test authentication to override the local account. A personal account or a service account can be used for this test.
  * You may need to grant access to the public schema for your new database user: `GRANT ALL ON SCHEMA public TO myaccount@example.com;`


### CI Platform Setup

Cloud Build is used to run tests against Google Cloud resources in test project: llama-index-cloud-sql-testing.
Each test has a corresponding Cloud Build trigger, see [all triggers][triggers].
These tests are registered as required tests in `.github/sync-repo-settings.yaml`.

#### Trigger Setup

Cloud Build triggers (for Python versions 3.9 to 3.11) were created with the following specs:

```YAML
name: pg-integration-test-pr-py39
description: Run integration tests on PR for Python 3.9
filename: integration.cloudbuild.yaml
github:
  name: llamaindex-cloud-sql-pg-python
  owner: googleapis
  pullRequest:
    branch: .*
    commentControl: COMMENTS_ENABLED_FOR_EXTERNAL_CONTRIBUTORS_ONLY
ignoredFiles:
  - docs/**
  - .kokoro/**
  - .github/**
  - "*.md"
substitutions:
  _DATABASE_ID: <ADD_VALUE>
  _INSTANCE_ID: <ADD_VALUE>
  _REGION: us-central1
  _VERSION: "3.9"
```

Use `gcloud builds triggers import --source=trigger.yaml` to create triggers via the command line

#### Project Setup

1. Create an Cloud SQL for PostgreSQL instance and database
1. Setup Cloud Build triggers (above)

#### Run tests with Cloud Build

* Run integration test:

    ```bash
    gcloud builds submit --config integration.cloudbuild.yaml --region us-central1 --substitutions=_INSTANCE_ID=$INSTANCE_ID,_DATABASE_ID=$DATABASE_ID,_REGION=$REGION
    ```

#### Trigger

To run Cloud Build tests on GitHub from external contributors, ie RenovateBot, comment: `/gcbrun`.

#### Code Coverage
Please make sure your code is fully tested. The Cloud Build integration tests are run with the `pytest-cov` code coverage plugin. They fail for PRs with a code coverage less than the threshold specified in `.coveragerc`.  If your file is inside the main module and should be ignored by code coverage check, add it to the `omit` section of `.coveragerc`.

Check for code coverage report in any Cloud Build integration test log.
Here is a breakdown of the report:
- `Stmts`:  lines of executable code (statements).
- `Miss`: number of lines not covered by tests.
- `Branch`: branches of executable code (e.g an if-else clause may count as 1 statement but 2 branches; test for both conditions to have both branches covered).
- `BrPart`: number of branches not covered by tests.
- `Cover`: average coverage of files.
- `Missing`: lines that are not covered by tests.

## Documentation

### LlamaIndex Integration Docs

Google hosts documentation on LlamaIndex's site for individual integration pages:
[Vector Stores][vs], [Document Store][docstore], [Index Store][indexstore], [Reader][reader], and [Chat Store][chatstore].

Currently, manual PRs are made to the [LlamaIndex GitHub repo](https://github.com/run-llama/llama_index).

### API Reference

#### Build the documentation
API docs are templated in the `docs/` directory.

To test locally, run: `nox -s docs`

The nox session, `docs`, is used to create HTML to publish to googleapis.dev
The nox session, `docfx`, is used to create YAML to publish to CGC.

#### Publish the documentation

The kokoro docs pipeline runs when a new release is created. See `.kokoro/` for the release pipeline.

[vs]: https://docs.llamaindex.ai/en/latest/examples/vector_stores/CloudSQLPgVectorStoreDemo/
[chatstore]: https://docs.llamaindex.ai/en/latest/module_guides/storing/chat_stores/#google-cloud-sql-for-postgresql-chatstore
[reader]: https://docs.llamaindex.ai/en/latest/examples/data_connectors/CloudSQLPgReaderDemo/
[docstore]: https://docs.llamaindex.ai/en/latest/examples/docstore/CloudSQLPgDocstoreDemo/
[indexstore]: https://docs.llamaindex.ai/en/latest/examples/docstore/CloudSQLPgDocstoreDemo/
[triggers]: https://pantheon.corp.google.com/cloud-build/triggers?e=13802955&project=llamaindex-cloud-sql-testing