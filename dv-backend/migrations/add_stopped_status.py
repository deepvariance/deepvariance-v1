"""
Migration to add 'stopped' status to Job and Model tables.
This allows proper semantic distinction between failed (error) and stopped (manual cancellation).
"""
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def migrate():
    """Add 'stopped' status to Job and Model table constraints"""
    print("Starting migration to add 'stopped' status...")
    
    # Get database URL
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://deepvariance:deepvariance@localhost:5432/deepvariance"
    )
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    try:
        # Check if tables exist first
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'job'
            );
        """)
        job_exists = cur.fetchone()[0]
        
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'model'
            );
        """)
        model_exists = cur.fetchone()[0]
        
        if not job_exists or not model_exists:
            print("✗ Tables do not exist yet. Run the application first to create tables.")
            return
        
        # Drop and recreate the Job table status constraint
        print("Updating Job table constraint...")
        cur.execute("""
            ALTER TABLE job DROP CONSTRAINT IF EXISTS valid_status;
        """)
        
        cur.execute("""
            ALTER TABLE job ADD CONSTRAINT valid_status 
            CHECK (status IN ('pending', 'running', 'completed', 'failed', 'stopped'));
        """)
        
        # Drop and recreate the Model table status constraint
        print("Updating Model table constraint...")
        cur.execute("""
            ALTER TABLE model DROP CONSTRAINT IF EXISTS valid_status;
        """)
        
        cur.execute("""
            ALTER TABLE model ADD CONSTRAINT valid_status 
            CHECK (status IN ('draft', 'queued', 'training', 'ready', 'active', 'failed', 'stopped'));
        """)
        
        conn.commit()
        print("✓ Migration completed successfully!")
        print("  - Job table now accepts 'stopped' status")
        print("  - Model table now accepts 'stopped' status")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    migrate()
