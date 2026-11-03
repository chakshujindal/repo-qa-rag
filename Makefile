.PHONY: install lint format test demo eval run clean

UV ?= uv
# Offline settings: deterministic hashing embedder + FakeLLM, so no API key is needed.
DEMO_ENV = REPOQA_HOME=.repoqa REPOQA_EMBEDDER=hashing REPOQA_LLM=fake
# Load .env for real runs when it exists.
ENV_FILE = $(if $(wildcard .env),--env-file .env)

install:
	$(UV) sync

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:
	$(UV) run pytest

demo:
	$(DEMO_ENV) $(UV) run repoqa index . --name repoqa
	$(DEMO_ENV) $(UV) run repoqa ask --repo repoqa "How are Python files split into chunks?"
	$(DEMO_ENV) $(UV) run repoqa ask --repo repoqa "How does hybrid retrieval combine BM25 and vector search?"

eval:
	$(DEMO_ENV) $(UV) run repoqa eval tests/fixtures/questions.yaml --details

run:
	$(UV) run $(ENV_FILE) repoqa serve

clean:
	rm -rf .repoqa .pytest_cache .ruff_cache .coverage htmlcov
