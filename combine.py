import os

# Set your base project directory
base_directory = r'G:\God_engine'  # Change this to your project root

# Set output file
output_file_path = 'Combined_God_engine.txt'

# File types you want to include (extend as needed)
include_extensions = [
    '.py', '.html', '.css', '.js', '.ts', '.json',
    '.env', '.txt', '.md', '.yml', '.yaml'
]

# Optional folders to exclude (e.g., static/media/migrations/venv)
exclude_dirs = {'__pycache__', 'static', 'media', 'migrations', 'venv', '.git', 'node_modules'}

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for root, dirs, files in os.walk(base_directory):
        # Exclude unwanted folders
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            if any(filename.endswith(ext) for ext in include_extensions):
                relative_path = os.path.relpath(filepath, base_directory)
                output_file.write(f"\n===== {relative_path} =====\n")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        output_file.write(f.read())
                except Exception as e:
                    output_file.write(f"[Error reading file: {e}]\n")
