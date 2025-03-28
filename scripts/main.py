import os
import subprocess


def main():
    # Define the project root directory
    # (this assumes the script is located in the scripts/ folder)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    main_path = os.path.join(project_root, "src", "among_them", "main.py")

    if not os.path.exists(main_path):
        raise FileNotFoundError(f"File does not exist: {main_path}")

    # Run the Streamlit app with the correct path
    subprocess.run(
        ["poetry", "run", main_path],
        check=True,
    )


if __name__ == "__main__":
    main()
