.PHONY: build start stop restart logs clean help

# Image name
IMAGE_NAME = ai_hedge_fund
CONTAINER_NAME = AI_Hedge_Fund

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Start the container
start:
	docker-compose up -d
	@echo "AI Hedge Fund is now running!"
	@echo "Access the web interface at: http://localhost:8080"

# Stop the container
stop:
	docker-compose down

# Restart the container
restart: stop start

# View logs
logs:
	docker-compose logs -f

# Clean up images and containers
clean:
	docker-compose down
	docker container prune -f
	docker image rm -f $(IMAGE_NAME)

# Display help information
help:
	@echo "Available commands:"
	@echo "  make build              - Build the Docker image"
	@echo "  make start              - Start the container"
	@echo "  make stop               - Stop the container"
	@echo "  make restart            - Restart the container"
	@echo "  make logs               - View container logs"
	@echo "  make clean              - Clean up Docker resources"
	@echo ""
	@echo "After starting, access the web interface at: http://localhost:8080"
	@echo "You will be prompted to enter your API key on first run."