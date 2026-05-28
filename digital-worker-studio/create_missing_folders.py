import os
import re

def extract_tree_from_md(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"```(?:bash|text)?\n(.*?)```", content, re.S)
    if not match:
        raise ValueError("No folder structure code block found in markdown.")
    return match.group(1)

def get_depth(line):
    """
    Depth is determined by how many '│' or spaces appear before the branch symbol.
    We count groups of 3 characters (│␣␣) as one level.
    """
    prefix = line.split("├")[0].split("└")[0]
    return prefix.count("│")

def clean_folder_name(line):
    # Remove tree characters and spaces around them
    cleaned = line.replace("├─", "").replace("└─", "").replace("│", "").strip()
    return cleaned.rstrip("/")

def parse_and_create(tree_text, project_root):
    lines = tree_text.splitlines()
    path_stack = []
    top_level = None

    for line in lines:
        if not line.strip().endswith("/"):
            continue

        folder_name = clean_folder_name(line)

        # Detect top-level folder
        if top_level is None:
            top_level = folder_name
            continue

        depth = get_depth(line)

        # Adjust stack to current depth
        path_stack = path_stack[:depth]
        path_stack.append(folder_name)

        full_path = os.path.join(project_root, *path_stack)

        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            print(f"[+] Created: {full_path}")
        else:
            print(f"[=] Exists:  {full_path}")

def main():
    md_file = r"D:\Mine\Learining\GenAI\python\the-learning-curve-labs\digital-worker-studio\docs\FOLDER_STRUCTURE.md"
    project_root = r"D:\Mine\Learining\GenAI\python\the-learning-curve-labs\digital-worker-studio"

    print(f"Creating folders inside: {project_root}")

    tree_text = extract_tree_from_md(md_file)
    parse_and_create(tree_text, project_root)

if __name__ == "__main__":
    main()
