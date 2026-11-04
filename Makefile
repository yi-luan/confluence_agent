.PHONY: install dev up down seed test lint fmt eval eval-retrieval

install:            ## 安裝開發環境
	pip install -e ".[dev]"

up:                 ## 啟動整套服務（API + Postgres + Prometheus + Grafana）
	docker compose up -d --build

down:
	docker compose down

seed:               ## 灌入範例員工資料與知識庫文件
	docker compose exec app python -m app.cli seed

dev:                ## 本機開發模式（需要先有 Postgres）
	uvicorn app.main:create_app --factory --reload

test:
	pytest

lint:
	ruff check . && ruff format --check .

fmt:
	ruff check --fix . && ruff format .

eval-retrieval:     ## 只評估檢索（不呼叫 LLM）
	python -m eval.run_eval --retrieval-only

eval:               ## 完整評估（會呼叫 LLM、產生費用）
	python -m eval.run_eval
