# This Makefile is a starting point and can be customized based on your needs.
# You can add/remove services, adjust ports, or integrate with Docker/Compose.

.PHONY: setup api jupyter mlflow start-all stop-all logs

# Create environment
setup:
	chmod +x init_env.sh
	./init_env.sh

# Run services individually (foreground)
api:
	.venv/bin/uvicorn main:app --reload --port 8000

jupyter:
	.venv/bin/jupyter lab --no-browser --port 8888

mlflow:
	.venv/bin/mlflow ui --host 127.0.0.1 --port 5000

# Run all services in background
start-all:
	mkdir -p logs run
	nohup .venv/bin/uvicorn main:app --reload --port 8000 > logs/api.log 2>&1 & echo $$! > run/api.pid
	nohup .venv/bin/jupyter lab --no-browser --port 8888 > logs/jupyter.log 2>&1 & echo $$! > run/jupyter.pid
	nohup .venv/bin/mlflow ui --host 127.0.0.1 --port 5000 > logs/mlflow.log 2>&1 & echo $$! > run/mlflow.pid
	@echo "Services started. Logs in ./logs"

# Stop all background services
stop-all:
	@if [ -f run/api.pid ]; then kill `cat run/api.pid` || true; rm -f run/api.pid; fi
	@if [ -f run/jupyter.pid ]; then kill `cat run/jupyter.pid` || true; rm -f run/jupyter.pid; fi
	@if [ -f run/mlflow.pid ]; then kill `cat run/mlflow.pid` || true; rm -f run/mlflow.pid; fi
	@echo "Services stopped."

# Tail logs
logs:
	tail -f logs/*.log
