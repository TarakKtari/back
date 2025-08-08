# My Python Application

This is a Python application that uses PostgreSQL as its database. The application is containerized using Docker and can be easily started with Docker Compose.

## Prerequisites

- Docker installed on your machine. [Get Docker](https://docs.docker.com/get-docker/)
- Docker Compose installed. [Get Docker Compose](https://docs.docker.com/compose/install/)

## Setup and Running the Application

1. **Build and start the containers**:

   Run the following command to build the Docker images and start the services defined in `docker-compose.yml`:

   ```bash
   docker-compose up -d
   ```

   This will:

   - Build the Docker image for the Python application.
   - Start the PostgreSQL database.
   - Start the Python application.

2. **Access the application**:

   Once the containers are up and running, the application should be accessible at:

   ```
   http://127.0.0.1:5001
   ```

   Replace `5001` with the port you've mapped if it's different.

3. **Stopping the application**:

   To stop the application, press `CTRL+C` in the terminal where `docker-compose` is running. Alternatively, you can stop the containers with:

   ```bash
   docker-compose down
   ```

4. **Viewing logs**:

   You can view the logs of your running services with:

   ```bash
   docker-compose logs
   ```

   To follow the logs in real-time, use:

   ```bash
   docker-compose logs -f
   ```

## Cleaning Up

To remove all Docker containers, networks, and volumes created by `docker-compose`, run:

```bash
docker-compose down -v
```
