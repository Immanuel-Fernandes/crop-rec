import os

# Set the PORT environment variable for Streamlit
port = int(os.environ.get("PORT", 8501))

# Run Streamlit
os.system(f"streamlit run main.py --server.port={port} --server.address=0.0.0.0")
