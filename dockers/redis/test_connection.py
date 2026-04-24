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
        print("\nConnection successful!")
        print(f"   Server PING response: {response}")

        # Perform a basic SET and GET to verify full functionality
        print("\n--- Performing basic operations ---")
        r.set('dev_test_key', 'it_works!')
        print("   - SET key 'dev_test_key' to 'it_works!'")

        value = r.get('dev_test_key')
        print(f"   - GET key 'dev_test_key' and received value: '{value}'")

        if value == 'it_works!':
            print("\nAll operations completed successfully! Your Redis instance is ready.")
        else:
            print("\nGET operation returned an unexpected value.")

    else:
        print("\nConnection appeared to succeed, but PING failed.")

except redis.exceptions.ConnectionError as e:
    print("\nConnection Failed. Check the following:")
    print("   1. Is the Docker container running and healthy? (Check with 'docker compose ps')")
    print(f"   Error details: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
