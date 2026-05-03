import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("DATABASE_URL not found in environment variables")
    exit(1)

print("Connecting to database...")
# Create a fresh connection with proper configuration
conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    prepare_threshold=None,  # Disable prepared statements to avoid conflicts
    row_factory=dict_row
)

print("Creating PostgresSaver...")
checkpointer = PostgresSaver(conn=conn)

print("Setting up checkpoint tables...")
try:
    checkpointer.setup()
except psycopg.errors.DuplicatePreparedStatement:
    # If prepared statement error, create new connection and retry
    print("Retrying with fresh connection...")
    conn.close()
    conn = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        prepare_threshold=None,
        row_factory=dict_row
    )
    checkpointer = PostgresSaver(conn=conn)
    checkpointer.setup()

print("Checkpoint tables created successfully!")
print("\nTables created:")
print("- checkpoints")
print("- checkpoint_writes") 
print("- checkpoint_blobs")

# Test listing checkpoints with proper config
print("\nTesting checkpoint listing...")
try:
    # List needs a proper config with thread_id
    config = {"configurable": {"thread_id": "test_thread"}}
    checkpoints = list(checkpointer.list(config))
    print(f"Found {len(checkpoints)} checkpoints")
except Exception as e:
    print(f"Note: {e}")
    print("This is normal if no checkpoints exist yet")

conn.close()
print("\nSetup completed successfully! Tables are ready for use.")