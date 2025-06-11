import os

def find_null_bytes(filepath):
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            if b'\x00' in content:
                positions = []
                for i, byte in enumerate(content):
                    if byte == 0:
                        positions.append(i)
                print(f"Null byte(s) found in: {filepath}")
                print(f"Positions: {positions[:10]}")  # Show first 10 positions
                return True
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return False

def walk_directory_and_check(start_dir):
    found_any = False
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.py'): # Only check Python files
                full_path = os.path.join(root, file)
                if find_null_bytes(full_path):
                    found_any = True
    if not found_any:
        print(f"No null bytes found in .py files under {start_dir}")

if __name__ == "__main__":
    # Path to your 'rag_book_ai' directory
    project_root = r"d:\maimoon\Vs Code\Book-AI-Application\Book-AI-Application\rag_book_ai"
    walk_directory_and_check(project_root)
