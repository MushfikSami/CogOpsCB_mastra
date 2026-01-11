# Graphiti - Neo4j - Gemini - Ingestion

## Table of Contents
- [Set up Neo4j with Docker](#set-up-neo4j-with-docker)
- [Connecting to Your Secure Neo4j Docker Instance ](#connecting-to-your-secure-neo4j-docker-instance)
- [Managing Graph Data in Your Secure Neo4j Docker Instance](#managing-graph-data-in-your-secure-neo4j-docker-instance) 



# Set up Neo4j with Docker

## Table of contents
  - [Step 1: Create Project Structure and Set Permissions](#step-1-create-project-structure-and-set-permissions)
  - [Step 2: Secure the Admin Password via Docker Secrets](#step-2-secure-the-admin-password-via-docker-secrets)
  - [Step 3: Generate a TLS Certificate for Your IP Address (with SAN)](#step-3-generate-a-tls-certificate-for-your-ip-address-with-san)
  - [Step 4: Configure Neo4j with Modern 5.x Settings](#step-4-configure-neo4j-with-modern-5x-settings)
  - [Step 5: Create the Docker Compose File. Launch, Verify](#step-5-create-the-docker-compose-file-launch-verify)


### Step 1: Create Project Structure and Set Permissions

This step organizes all your configuration and data files and ensures the Neo4j process inside the container can access them without permission errors.



```bash
# Create the complete directory structure
mkdir -p neo4j-local-secure/{conf,data,logs,plugins,certificates,secrets,exports}
cd neo4j-local-secure

# Set ownership to match the container's internal neo4j user (UID 101, GID 101)
sudo chown -R 101:101 data logs conf plugins certificates
```
**Why this is important:** This prevents "Permission Denied" errors when the non-root `neo4j` user in the container tries to write to the data or log directories.

---

### Step 2: Secure the Admin Password via Docker Secrets

We will generate a strong password and store it in a file. Docker Secrets will manage this file, keeping the password out of your configuration and environment variables.


####  Without root permissions 

```bash
# 1. Generate a strong, random password for the 'neo4j' user
openssl rand -base64 32 > secrets/neo4j_password

# 2. Create the final secret file in the format Neo4j expects: username/<password>
echo "neo4j/$(cat secrets/neo4j_password)" > secrets/neo4j_auth

# 3. (Optional) Display the password once to save it in your password manager
echo "Your secure password is: $(cat secrets/neo4j_password)"
```
**Why this is important:** This is a secure method for handling credentials, vastly superior to hardcoding them or using plain environment variables.

---
####  With root permissions 
**IMPORTANT**: If you want to do it with root:The problem is a combination of file permissions in the root (`/`) directory and a common issue with how `sudo` works with output redirection (`>`).

Here is the explanation and the corrected set of commands for only this step.

### The Problem

1.  **Directory Ownership:** When you ran `sudo mkdir -p ...`, the `secrets` directory was created and is owned by the `root` user.
2.  **Redirection (`>`) Failure:** The command `sudo openssl ... > ...` fails because the shell (bash) tries to set up the file redirection *before* it runs the `sudo` command. Your regular user (`your non root user`) does not have permission to write to the `root`-owned `secrets` directory, so the operation fails before `openssl` even runs.

### The Solution

We need to pipe (`|`) the output of `openssl` to a command that can be run with `sudo` and can write to a file. The `tee` command is perfect for this.

Here are the corrected commands to create your password secrets in `/neo4j-local-secure/secrets`.

```bash
# Ensure you are in the correct directory
cd /neo4j-local-secure

# --- CORRECTED COMMANDS START HERE ---

# 1. Generate the password and pipe it to `tee`, which uses sudo to write the file.
openssl rand -base64 32 | sudo tee secrets/neo4j_password > /dev/null

# 2. Create the final auth secret file using the same technique.
echo "neo4j/$(sudo cat secrets/neo4j_password)" | sudo tee secrets/neo4j_auth > /dev/null

# 3. (Optional) Verify the files were created and are owned by root.
sudo ls -l secrets

# 4. (Optional) Display the password so you can save it.
echo "Your secure password is: $(sudo cat secrets/neo4j_password)"
```

#### Command Explanation

*   **`openssl rand -base64 32`**: This part is the same; it generates the random password string.
*   **`|`**: This is the "pipe" operator. It sends the output of the command on its left as the input to the command on its right.
*   **`sudo tee secrets/neo4j_password`**:
    *   `tee` is a command that reads input and writes it to both the screen and a file.
    *   By running `sudo tee`, the `tee` process itself has root privileges and therefore has permission to write the file `secrets/neo4j_password`.
*   **`> /dev/null`**: We add this because `tee` also outputs to the screen by default. Piping the screen output to `/dev/null` (a special file that discards everything written to it) keeps your terminal clean.

---

### Step 3: Generate a TLS Certificate for Your IP Address (with SAN)

Modern browsers and clients require the IP address to be in the **Subject Alternative Name (SAN)** field of a certificate. This command creates a certificate that is valid for your specific server IP.

#### For Remote Server
```bash
# Set your server's IP address as an environment variable
export SERVER_IP="YOUR_SERVER_IP"

# Generate the key and certificate with the IP in both the CN and SAN fields
# This single command works on most modern systems (OpenSSL ≥ 1.1.1)
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -subj "/CN=${SERVER_IP}" \
  -addext "subjectAltName = IP:${SERVER_IP}" \
  -keyout certificates/private.key \
  -out certificates/public.crt
```
**Why this is important:** Including the IP in the SAN is the modern standard and prevents hostname mismatch errors from clients, ensuring a secure and valid TLS connection.

#### For localhost

```bash
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -subj "/CN=localhost" \
  -addext "subjectAltName = DNS:localhost" \
  -keyout certificates/private.key \
  -out certificates/public.crt
```

* **Use sudo if using root (obviously)**

* give access: ```sudo chown -R 101:101 certificates```

---

### Step 4: Configure Neo4j with Modern 5.x Settings

Create the file `conf/neo4j.conf`. This configuration enforces encrypted-only connections and advertises the correct IP address to connecting clients.

**Remember to replace `YOUR_SERVER_IP` in this file, if you are not using localhost**

```ini
# conf/neo4j.conf (Corrected for Neo4j 5.x)

# ================= Network =================
# Listen on all interfaces inside the container
server.default_listen_address=0.0.0.0

# Advertise the correct LAN IP to clients. CRITICAL for drivers and clustering.
# For a LAN setup, uncomment and set this. For a pure localhost setup, leave it commented.
# server.default_advertised_address=YOUR_SERVER_IP

# ================= Connectors =================
# Disable the insecure HTTP connector entirely.
server.http.enabled=false

# --- HTTPS Connector ---
server.https.enabled=true
server.https.listen_address=:7473
# TLS configuration using SSL policy
dbms.ssl.policy.https.enabled=true
dbms.ssl.policy.https.base_directory=/certificates
dbms.ssl.policy.https.private_key=private.key
dbms.ssl.policy.https.public_certificate=public.crt

# --- Bolt Connector ---
server.bolt.enabled=true
server.bolt.listen_address=:7687
# Enforce encrypted connections for Bolt.
server.bolt.tls_level=REQUIRED
# Use the same SSL policy for Bolt
dbms.ssl.policy.bolt.enabled=true
dbms.ssl.policy.bolt.base_directory=/certificates
dbms.ssl.policy.bolt.private_key=private.key
dbms.ssl.policy.bolt.public_certificate=public.crt

# ================= Security =================
# Keep strict validation enabled. This is a critical security feature.
server.config.strict_validation.enabled=true

# ================= Memory (Tune for your machine) =================
server.memory.heap.initial_size=4G
server.memory.heap.max_size=4G
server.memory.pagecache.size=8G
```

#### Neo4j Memory Configuration Breakdown

Neo4j uses memory in two primary ways: the **JVM heap** for runtime operations and the **page cache** for caching graph data to reduce disk I/O. The configuration you provided is from Neo4j’s configuration file (likely `neo4j.conf`):

1. **server.memory.heap.initial_size=4G**
   - This sets the **initial JVM heap size** to 4GB.
   - The heap is used for storing graph metadata, query execution, and transactional state during Neo4j’s operation.

2. **server.memory.heap.max_size=4G**
   - This sets the **maximum JVM heap size** to 4GB.
   - Since the initial and maximum heap sizes are equal, the heap is fixed at 4GB, preventing resizing and stabilizing memory usage for the JVM.

3. **server.memory.pagecache.size=8G**
   - This sets the **page cache size** to 8GB.
   - The page cache in Neo4j is used to store the graph’s data (nodes, relationships, and properties) in memory, reducing disk access for faster query performance.

### RAM Requirements for Neo4j

To run Neo4j with this configuration, you need enough RAM to accommodate the heap, page cache, and additional overhead for the operating system and Neo4j’s internal processes. Here’s the breakdown:

1. **JVM Heap**: 
   - Fixed at 4GB for runtime operations.

2. **Page Cache**: 
   - Set to 8GB for caching graph data.

3. **Additional Overhead**:
   - **JVM Overhead**: Beyond the heap, Neo4j requires memory for metaspace (class metadata), thread stacks, and garbage collection. This typically adds ~10–20% of the heap size, so estimate **0.5–1GB**.
   - **Operating System**: The OS needs memory for its processes, file system buffers, and other tasks. Reserve **1–2GB** for a server running Neo4j.
   - **Neo4j Native Memory**: Neo4j may use additional off-heap memory for internal buffers and indexing, which could add another **0.5–1GB** depending on the workload.

4. **Total RAM Estimate**:
   - Heap: 4GB
   - Page Cache: 8GB
   - JVM Overhead: ~0.5–1GB
   - OS: ~1–2GB
   - Neo4j Native Memory: ~0.5–1GB
   - **Minimum Total**: 4 + 8 + 0.5 + 1 + 0.5 = **14GB**
   - **Recommended Total**: To ensure smooth performance and account for variability (e.g., spikes in memory usage), aim for **16–24GB of RAM**. For production systems, 32GB is safer if you expect moderate to heavy workloads or additional services running on the same machine.

### SSD Requirements for Neo4j

The SSD requirements depend on the size of your graph database and how Neo4j uses the page cache:

1. **Page Cache and Disk I/O**:
   - The 8GB page cache is used to store frequently accessed portions of the graph (nodes, relationships, indexes) in memory to minimize disk reads.
   - In Neo4j, the on-disk data size is typically much larger than the page cache. A rough rule of thumb is that the page cache should be **10–50% of the total database size** for optimal performance, depending on the workload (read-heavy vs. write-heavy).

2. **Estimating Disk Size**:
   - If the page cache is 8GB, the on-disk database size could range from **16GB to 80GB** or more, depending on how much of the graph needs to be cached for performance.
     - For example, if you want 50% of the database in the page cache, an 8GB cache suggests a ~16GB database.
     - For read-heavy workloads, you might need a smaller cache relative to the database size (e.g., 10%), implying a database size of ~80GB.
   - Neo4j’s storage includes data files, indexes, and transaction logs, so plan for **at least 2–3x the page cache size** as a starting point (e.g., **16–24GB**).
   - For production systems with growing datasets, **100GB or more** of SSD storage is common.

3. **Performance Considerations**:
   - **SSDs are critical** for Neo4j because graph traversals and queries can be I/O-intensive, especially if the page cache cannot hold the working set of data.
   - Use high-performance SSDs (e.g., NVMe) with good random read/write performance for better query latency.
   - Keep 20–30% free space on the SSD for wear leveling and to maintain performance.

### Recommendations for Your Setup

- **Minimum RAM**: **16GB** to cover the 4GB heap, 8GB page cache, and overhead for the OS and Neo4j. For production or heavier workloads, **24–32GB** is recommended to avoid memory contention.
- **Minimum SSD**: **32–64GB** for small to medium databases (e.g., a few million nodes and relationships). For larger graphs or production use, plan for **100GB–500GB** or more, depending on your data size.
- **Tuning Tips**:
  - **Heap Size**: A 4GB heap is reasonable for small to medium workloads. For larger graphs or high-concurrency queries, you might need to increase it (e.g., 8GB) if you have more RAM.
  - **Page Cache**: The 8GB page cache is good for databases up to ~16–80GB, depending on access patterns. If your database grows significantly larger, consider increasing the page cache or RAM.
  - **Monitoring**: Use Neo4j’s monitoring tools (e.g., metrics or logs) to check page cache hit ratios and heap usage. If the cache hit ratio is low, increase the page cache size or RAM.
  - **OS**: Use a 64-bit OS (e.g., Linux, Ubuntu, or CentOS) for Neo4j, as 32-bit systems can’t handle large memory allocations effectively.
- **Workload Context**: If you can share the expected number of nodes, relationships, or query types (e.g., read-heavy vs. write-heavy), I can refine the SSD estimate further.

### Example Hardware Setup
- **Small Setup (Development)**: 16GB RAM, 64GB SSD, single-socket CPU (4–8 cores).
- **Production Setup (Medium Workload)**: 32GB RAM, 256GB NVMe SSD, 8–16 core CPU for handling concurrent queries.


---

### Step 5: Create the Docker Compose File. Launch, Verify

Create the `docker-compose.yml` file. This is the heart of the deployment, binding the service specifically to your LAN IP and using the robust `cypher-shell` healthcheck.

**Remember to replace `YOUR_SERVER_IP` in this file.**

```yaml

services:
  neo4j:
    image: neo4j:5-enterprise          # Tracks the latest 5.x version
    container_name: neo4j_local_secure
    restart: unless-stopped
    user: "101:101"                    # Run as a non-root user

    # Security: Bind host ports specifically to your LAN IP, not 0.0.0.0 (all interfaces).
    ports:
      - "YOUR_SERVER_IP:7473:7473"     # Secure HTTPS
      - "YOUR_SERVER_IP:7687:7687"     # Secure Bolt

    volumes:
      - ./data:/data
      - ./logs:/logs
      - ./conf:/conf
      - ./plugins:/plugins
      - ./certificates:/certificates
      - ./exports:/exports

    environment:
      - NEO4J_AUTH_FILE=/run/secrets/neo4j_auth_secret
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes

    secrets:
      - source: neo4j_auth_secret
        target: neo4j_auth_secret

    healthcheck:
      # Robust check using cypher-shell to query the database directly.
      # 'bolt+ssc' scheme trusts the self-signed certificate for the healthcheck.
      test: ["CMD-SHELL", "PASS=$$(cut -d/ -f2 /run/secrets/neo4j_auth_secret) && cypher-shell -a bolt+ssc://localhost:7687 -u neo4j -p \"$$PASS\" 'RETURN 1' >/dev/null 2>&1 || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 10

secrets:
  neo4j_auth_secret:
    file: ./secrets/neo4j_auth
```

#### For localhost:

```bash
    # Security: Bind host ports specifically to localhost (127.0.0.1).
    ports:
      - "127.0.0.1:7473:7473"     # Accessible only from https://localhost:7473
      - "127.0.0.1:7687:7687"     # Accessible only from bolt+ssc://localhost:7687
```


You are now ready to start the service and connect.

```bash
# 1. Start the service in the background
docker compose up -d

# 2. Check the status. Wait for the STATUS to become 'running (healthy)'
docker compose ps

# 3. (Optional) Follow the logs on first startup
docker logs -f neo4j_local_secure
```


### Harden with a Firewall (Recommended)

For true defense-in-depth, configure a firewall on the host machine to only allow traffic to the Neo4j ports from your trusted local network.

**UFW (Ubuntu) Example:**

```bash
# Replace with your LAN subnet (e.g., 192.168.1.0/24)
export LAN_SUBNET="YOUR_LAN_SUBNET"

# Allow incoming connections from the LAN to the Neo4j ports
sudo ufw allow from ${LAN_SUBNET} to any port 7473 proto tcp
sudo ufw allow from ${LAN_SUBNET} to any port 7687 proto tcp

# Ensure UFW is enabled and check the status
sudo ufw enable
sudo ufw status
```
**Why this is important:** Even if Docker is bound to a specific IP, the firewall acts as a second, powerful layer of defense against unwanted network access.


# Connecting to Your Secure Neo4j Docker Instance 

This guide provides detailed instructions on how to connect to and verify your secure Neo4j 5.x Docker container. It covers all common scenarios, including connecting from the host machine, a remote machine on the same network, and securely through an SSH tunnel.

## Prerequisites

Before you begin, ensure you have the following information and tools:

1.  **Server IP Address**: The LAN IP address of the machine hosting the Docker container (e.g., `192.168.1.101`). This will be referred to as `YOUR_SERVER_IP`.
2.  **Neo4j Password**: The secure password you generated. You can retrieve it from the host machine at any time by running:
    ```bash
    # Run this from your neo4j-local-secure directory
    sudo cat secrets/neo4j_password
    ```
3.  **`cypher-shell`**: The official Neo4j command-line interface. This must be installed on any machine you wish to connect from.

---

**How to Connect:**

*   **Neo4j Browser:** Navigate to `https://YOUR_SERVER_IP:7473`
    *   *You will see a security warning. This is expected. Click "Advanced" and proceed.*
*   **Drivers & Tools:** Use a secure connection string. The `+ssc` scheme is a convenient shortcut that trusts self-signed certificates without needing a custom trust store.
    *   `bolt+ssc://YOUR_SERVER_IP:7687`
*   **`cypher-shell` from your host machine:**
    ```bash
    cypher-shell -a bolt+ssc://YOUR_SERVER_IP:7687 -u neo4j -p "$(cat secrets/neo4j_password)"
    ```


#### For Localhost: 
**Local Port Forwarding** (or an "SSH Tunnel").

you bound the ports to `127.0.0.1` on the server, you cannot connect to `https://YOUR_SERVER_IP:7473`. 

The solution is to tell SSH to create a secure tunnel from your local machine, through the SSH connection, directly to the `localhost` port on the remote server.

### The Concept

You will forward a port on your **local machine** (the one you are typing on) to the Neo4j port on the **remote server's localhost**.

*   Traffic going into `localhost:7473` on **your laptop**
*   ...travels securely through the SSH connection...
*   ...and comes out on `localhost:7473` on the **remote server**, where Neo4j is listening.

### The Solution: The `-L` Flag in SSH

You need to forward two ports: `7473` for the browser (HTTPS) and `7687` for the Bolt driver. You can do this in a single SSH command by using the `-L` flag twice.

#### Disconnect and Reconnect with Port Forwarding

First, if you are currently connected to your server, `exit` that SSH session.

Now, reconnect using the following command. This is the **only command you need to change**.

```bash
# General Syntax: ssh -L <local_port>:<destination_host>:<destination_port> user@server
#
# We will forward:
# - Our local port 7473 to the remote server's localhost:7473
# - Our local port 7687 to the remote server's localhost:7687

ssh -L 7473:localhost:7473 -L 7687:localhost:7687 user@xxx.xxx.xxx.xxx
```

**Command Breakdown:**

*   **`ssh user@the.server.ip.addreess`**: Your standard SSH login command.
*   **`-L 7473:localhost:7473`**:
    *   `-L`: Specifies **L**ocal port forwarding.
    *   `7473`: The port to open on **your local machine**.
    *   `localhost`: The destination host *from the remote server's perspective*. We want to connect to `localhost` on the server.
    *   `7473`: The destination port on the remote server.
*   **`-L 7687:localhost:7687`**: Does the same thing for the Bolt port.

#### Keep the SSH Connection Open

As long as this SSH terminal window is open, the secure tunnels are active. If you close this window, the tunnels will close.

#### Connect Using `localhost` on Your Local Machine

Now, on your local machine (your laptop), you can access Neo4j as if it were running locally.

*   **Open Your Web Browser (on your laptop):**
    Navigate to: `https://localhost:7473`
    *   Your browser will send the request to your local port 7473.
    *   SSH will intercept it, send it through the tunnel, and deliver it to Neo4j on the server.
    *   You will still see the security warning for the self-signed certificate, which is expected.

*   **Use `cypher-shell` or a Driver (from your laptop):**
    Use the connection string: `bolt+ssc://localhost:7687`
    ```bash
    # You would run this in a NEW terminal window on your local machine,
    # NOT in the SSH window.
    cypher-shell -a bolt+ssc://localhost:7687 -u neo4j -p "YOUR_SECURE_PASSWORD"
    ```

This is the standard and most secure way to manage services that are intentionally not exposed to the network. You maintain a very high level of security on the server while still having full access for management from your trusted machine.

---

## Connection Methods

The correct connection method depends on how you configured the `ports` section in your `docker-compose.yml` file.

### Scenario 1: Ports are Bound to Your Server's LAN IP

This is the standard setup for making Neo4j available to other machines on your local network. Your `docker-compose.yml` `ports` section looks like this:

```yaml
    ports:
      - "YOUR_SERVER_IP:7473:7473" # Secure HTTPS
      - "YOUR_SERVER_IP:7687:7687" # Secure Bolt
```

#### ► Method 1: Connecting from the Host Machine Terminal

This is the most direct way to interact with the database.

1.  **Open a terminal** on the machine running the Docker container.
2.  **Navigate** to your `neo4j-local-secure` project directory.
3.  **Run the `cypher-shell` command**:

    ```bash
    cypher-shell \
      -a "bolt+ssc://YOUR_SERVER_IP:7687" \
      -u neo4j \
      -p "$(sudo cat secrets/neo4j_password)"
    ```
    > **Note**: The `+ssc` in the address tells the client to trust the server's **S**elf-**S**igned **C**ertificate, which is exactly what we need for this setup.

4.  **Verify the connection** as described in the [Verification](#verification) section below.

#### ► Method 2: Connecting from a Remote Machine Terminal

You can connect from any other computer on the same LAN that has `cypher-shell` installed.

1.  **Open a terminal** on your remote machine (e.g., your laptop).
2.  **Run the `cypher-shell` command**. You will be prompted to enter your password securely.

    ```bash
    cypher-shell -a "bolt+ssc://YOUR_SERVER_IP:7687" -u neo4j
    ```
3.  **Enter your password** when prompted.
4.  **Verify the connection** as described in the [Verification](#verification) section.

#### ► Method 3: Connecting via Web Browser

You can access the Neo4j Browser from any machine on the same LAN.

1.  **Open your web browser** (Chrome, Firefox, etc.).
2.  **Navigate to the following URL**:
    `https://YOUR_SERVER_IP:7473`
3.  **Handle the Security Warning**: Your browser will show a privacy or security warning because the certificate is self-signed. This is expected.
    *   Click **"Advanced"**.
    *   Click **"Proceed to YOUR_SERVER_IP (unsafe)"** or "Accept the Risk and Continue".
4.  **Log in**: Use `neo4j` as the username and your secure password. The connection URI should default to `bolt://YOUR_SERVER_IP:7687`.
5.  Click **Connect**.

---

### Scenario 2: Ports are Bound to `localhost` (127.0.0.1)

This is a high-security setup where the database is not exposed to the network at all. Remote access is only possible via a secure SSH tunnel. Your `docker-compose.yml` `ports` section looks like this:

```yaml
    ports:
      - "127.0.0.1:7473:7473"
      - "127.0.0.1:7687:7687"
```

#### ► Method 1: Connecting from the Host Machine (Terminal & Browser)

Connecting from the host is simple because you are already on `localhost`.

*   **Terminal**:
    ```bash
    cypher-shell \
      -a "bolt+ssc://localhost:7687" \
      -u neo4j \
      -p "$(sudo cat secrets/neo4j_password)"
    ```
*   **Browser**:
    1.  Navigate to `https://localhost:7473`.
    2.  Bypass the security warning as explained above.
    3.  Log in with your credentials.

#### ► Method 2: Connecting from a Remote Machine (via SSH Tunnel)

This is the standard, secure way to manage a service that is not exposed to the network.

1.  **Establish the SSH Tunnel**: From your remote machine's terminal, run the following command to connect to your server. This command forwards your local ports `7473` and `7687` through the SSH connection to the server's `localhost`.

    ```bash
    # Syntax: ssh -L <local_port>:<destination_host>:<destination_port> user@server
    ssh -L 7473:localhost:7473 -L 7687:localhost:7687 your_user@YOUR_SERVER_IP
    ```
    **Important**: You must keep this SSH terminal window open. The tunnel remains active only as long as this connection is open.

2.  **Connect in a *New* Terminal or Browser**: With the tunnel active, open a **new** terminal window or your browser on your remote machine and connect to `localhost` as if Neo4j were running locally.

    *   **New Terminal**:
        ```bash
        # Connect to your local port, which SSH will forward to the server
        cypher-shell -a "bolt+ssc://localhost:7687" -u neo4j
        ```
    *   **Browser**:
        1.  Navigate to `https://localhost:7473`.
        2.  Bypass the security warning.
        3.  Log in. Your traffic will be securely tunneled to the server.

---

## Verification

A successful connection can be confirmed in two ways.

### In `cypher-shell`

Upon successful connection, your terminal prompt will change to:
`neo4j@bolt+ssc://...> `

Run this basic command to confirm the database is responsive:

```cypher
SHOW DATABASES;
```

A successful query will return a table of the available databases:

```
+--------------------------------------------------------------------------------------------+
| name     | type       | aliases | access      | address               | role      | writer |
+--------------------------------------------------------------------------------------------+
| "neo4j"  | "standard" | []      | "read-write"| "localhost:7687"      | "primary" | TRUE   |
| "system" | "system"   | []      | "read-write"| "localhost:7687"      | "primary" | TRUE   |
+--------------------------------------------------------------------------------------------+
```

To exit the shell, type `:exit` and press Enter.

### In Neo4j Browser

After logging in, you will see the main Neo4j Browser interface. The top-left corner will show that you are connected to the database, and you can type Cypher queries into the editor at the top of the screen.


---

# Managing Graph Data in Your Secure Neo4j Docker Instance

This guide provides essential instructions for managing the data within your Neo4j database. It covers how to perform logical exports of your entire graph into portable formats and how to completely clear the database of all nodes and relationships for a clean reset.

These operations are performed while the database is running and do not require you to stop the container.

## Table of Contents
- [Prerequisite: Installing the APOC Plugin](#prerequisite-installing-the-apoc-plugin)
- [Exporting All Graph Data](#exporting-all-graph-data)
- [Deleting All Graph Data](#deleting-all-graph-data)

---

## Prerequisite: Installing the APOC Plugin

The most powerful and flexible way to manage data is with the **APOC ("Awesome Procedures On Cypher")** library. This is a one-time setup.

1.  **Download APOC**: Go to the [APOC Releases page](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases) and find the release that **exactly matches your Neo4j version**. For example, if you are using Neo4j `5.26.10`, download the APOC `5.26.10` JAR file.
    *   Under the "Assets" section, download the file named `apoc-x.x.x-core.jar`.

2.  **Place the Plugin**: Move the downloaded `.jar` file into the `./plugins` directory of your project on the host machine.
    ```bash
    # Example command
    mv ~/Downloads/apoc-5.26.10-core.jar ./plugins/
    ```

3.  **Configure Neo4j to Allow APOC**: Edit your `conf/neo4j.conf` file and add the following line. This gives APOC the necessary permissions to perform operations like writing files.

    ```ini
    # Add this line to the end of your conf/neo4j.conf
    dbms.security.procedures.unrestricted=apoc.*
    ```

4.  **Restart the Container**: To load the plugin and apply the new configuration, restart your Docker service from your project directory.
    ```bash
    docker compose down && docker compose up -d
    ```

---

## Exporting All Graph Data

These procedures export your data into files saved on the host machine, making them easy to access, share, or use for migration.

#### Setup: Create a Shared Directory for Exports

To easily retrieve exported files, we'll map a local directory into the container.

1.  **Create an `exports` directory** on your host:
    ```bash
    mkdir ./exports
    ```
2.  **Add the volume to `docker-compose.yml`**:
    ```yaml
    # in your docker-compose.yml
    services:
      neo4j:
        # ... other settings
        volumes:
          - ./data:/data
          - ./logs:/logs
          - ./conf:/conf
          - ./plugins:/plugins
          - ./certificates:/certificates
          - ./exports:/exports   # <--- ADD THIS LINE
    ```
3.  **Restart the container** if you just added the volume: `docker compose up -d --force-recreate`.

#### ► Export Option 1: Cypher Script
This method creates a single `.cypher` file containing all the `CREATE` statements needed to perfectly rebuild your graph. It is ideal for backups and migrations to other Neo4j instances.

Connect to your database via `cypher-shell` or the Neo4j Browser and run:
```cypher
/*
  This procedure will create a 'all-data.cypher' file
  inside the ./exports directory on your host machine.
*/
CALL apoc.export.cypher.all('all-data.cypher', {
  format: 'cypher-shell',
  useOptimizations: {type: 'UNWIND', unwindBatchSize: 100}
});
```

#### ► Export Option 2: GraphML
GraphML is a standard XML-based format that can be imported into other graph visualization and analysis tools, such as Gephi.

```cypher
/*
  This procedure will create a 'all-data.graphml' file
  inside the ./exports directory on your host machine.
*/
CALL apoc.export.graphml.all('all-data.graphml', {});
```

---

## Deleting All Graph Data

This action removes **all nodes and relationships** from your `neo4j` database. It does **not** remove your user accounts, and it leaves your schema (indexes and constraints) intact.

#### ► Method 1: Simple Delete (For Small to Medium Graphs)
This command is easy to remember and effective for databases that are not excessively large.

```cypher
MATCH (n) DETACH DELETE n;
```
> **Warning**: On very large graphs (millions of nodes/edges), this single transaction can consume a large amount of memory and may fail. For large datasets, use the batched method below.

#### ► Method 2: Batched Delete (For Very Large Graphs)
This is the recommended, robust method for clearing any size of graph. It uses an APOC procedure to delete nodes in smaller, manageable batches, preventing memory issues.

```cypher
/*
  This procedure finds all nodes and runs DETACH DELETE on them in
  batches of 50,000 until the database is empty.
*/
CALL apoc.periodic.iterate(
  'MATCH (n) RETURN n',
  'DETACH DELETE n',
  {batchSize: 50000}
)
YIELD batches, total;
```

#### Verification
After running a delete command, you can confirm the database is empty with the following query. The expected result is `0`.
```cypher
MATCH (n) RETURN count(n);
```