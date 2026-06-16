import streamlit.web.cli as stcli
import sys

def main():
    sys.argv = ["streamlit", "run", "2026FWA.py", "--server.port=8501"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()