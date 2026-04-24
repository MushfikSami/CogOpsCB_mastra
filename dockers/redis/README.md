### **Easy, Conflict-Free Redis Deployment with Docker (No Admin Access Required)**

**Objective:**
To create a simple, persistent, and network-conflict-free Redis database for local development. This guide is designed for users who do not have administrative (`sudo`) privileges.

**Key Features of This Setup:**
*   **No `sudo` Required:** All commands are run as a regular user.
*   **No Root Directories:** All files are stored safely within your user's home directory.
*   **Conflict-Free Networking:** Uses a custom `10.x.x.x` IP address range to prevent common conflicts.
*   **No Password:** The server is configured for easy local development without authentication.
*   **Persistent Data:** Your Redis data is saved on your computer and will survive container restarts.

**Prerequisites:**
1.  **Docker and Docker Compose are installed.**
2.  **Your user account is part of the `docker` group.** (This is the standard way to allow non-root users to run Docker. If you get a "permission denied" error when running `docker ps`, ask a system administrator to run `sudo usermod -aG docker $USER` and then log out and back in).

---

### **Step 1: Create the Project Directory Structure**

We will create a clean folder structure inside your home directory to hold the configuration and data for your Redis instance.

1.  **Open your terminal and run this command to create the directories:**
    The `~` is a shortcut for your home directory (e.g., `/home/your_username`).
    ```bash
    mkdir -p ~/database/redis-local/data
    ```

2.  **Navigate into your new project directory.** All subsequent commands must be run from this location.
    ```bash
    cd ~/database/redis-local
    ```

---

### **Step 2: Create the Docker Compose Configuration File**

This is the main instruction file that tells Docker how to set up and run your Redis container, including the conflict-free network configuration.

1.  **Create a new file named `docker-compose.yml`** inside your `~/database/redis-local` directory.

2.  **Copy and paste the entire block of code below** into that `docker-compose.yml` file.

    ```yaml
    # ~/database/redis-local/docker-compose.yml
    # A simple, root-free Redis setup with a custom network to prevent conflicts.

    version: '3.8'

    services:
      redis:
        # Use the official lightweight Redis image
        image: redis:7-alpine
        container_name: redis_local_dev
        restart: unless-stopped

        # Security: Binds the Redis port 6379 to your machine's 'localhost' only.
        # This prevents any other computer on your network from accessing it.
        ports:
          - "127.0.0.1:6379:6379"

        # Data Persistence: Maps the 'data' folder to the container's data directory.
        # This ensures your data is saved on your computer.
        volumes:
          - ./data:/data

        # Health Check: Docker will use this to ensure Redis is running correctly.
        healthcheck:
          test: ["CMD", "redis-cli", "ping"]
          interval: 10s
          timeout: 5s
          retries: 5

        # Conflict-Free Network: Connects this service to our custom network.
        networks:
          - redis-net

    # Custom Network Definition to Avoid Conflicts
    # Creates a dedicated network to prevent issues with VPNs or other services.
    networks:
      redis-net:
        driver: bridge
        ipam:
          driver: default
          config:
            # We define a subnet in the 10.x.x.x range, which is very unlikely to conflict.
            - subnet: "10.56.0.0/24"
    ```

---

### **Step 3: Launch and Verify the Service**

Now you can start your Redis server.

1.  **Stop any old versions (if you ran this before):**
    If this is your first time, you can skip this command. If you are re-running the setup, this ensures a clean start.
    ```bash
    docker compose down
    ```

2.  **Start the service in the background:**
    Make sure you are still in the `~/database/redis-local` directory. Docker will download the Redis image on the first run.
    ```bash
    docker compose up -d
    ```

3.  **Check the container status:**
    ```bash
    docker compose ps
    ```
    You should see a container named `redis_local_dev`. Wait for its `STATUS` to show `(healthy)`. This should only take a few seconds.

4.  **(Optional) Verify the Network:** You can confirm that your container is using the new IP range with this command:
    ```bash
    # The network name is a combination of the directory name ('redis-local') and the network name from the file.
    docker network inspect redis-local_redis-net
    ```
    In the output, look for the `IPAM` section. You will see your custom `"Subnet": "10.56.0.0/24"` listed, confirming that the conflict-avoidance strategy is active.

---

### **Step 4: Connect to Your Redis Instance with Python**

The final step is to run a Python script to confirm that your Redis instance is working correctly.

1.  **Install the required Python library:**
    ```bash
    pip install redis
    ```

2.  **Create a new file named `test_connection.py`** .

3.  **Copy and paste the Python code below** into your `test_connection.py` file.

    ```python
    import redis

    print("Attempting to connect to local Redis server...")

    try:
        # The connection requires no password and no SSL.
        # 'decode_responses=True' makes the output a normal string instead of bytes.
        r = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )

        # The 'ping' command is the simplest way to check if the connection is alive.
        response = r.ping()

        if response:
            print("\n✅ Connection successful!")
            print(f"   Server PING response: {response}")

            # Perform a basic SET and GET to verify full functionality
            print("\n--- Performing basic operations ---")
            r.set('dev_test_key', 'it_works!')
            print("   - SET key 'dev_test_key' to 'it_works!'")

            value = r.get('dev_test_key')
            print(f"   - GET key 'dev_test_key' and received value: '{value}'")

            if value == 'it_works!':
                print("\n✅ All operations completed successfully! Your Redis instance is ready.")
            else:
                print("\n❌ GET operation returned an unexpected value.")

        else:
            print("\n❌ Connection appeared to succeed, but PING failed.")

    except redis.exceptions.ConnectionError as e:
        print("\n❌ Connection Failed. Check the following:")
        print("   1. Is the Docker container running and healthy? (Check with 'docker compose ps')")
        print(f"   Error details: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    ```

4.  **Run the script from your terminal:**
    ```bash
    python test_connection.py
    ```

If everything is set up correctly, you will see a **"Connection Successful!"** message, followed by confirmation that the SET and GET operations worked. Your local, persistent, and conflict-free Redis instance is now ready to use.
