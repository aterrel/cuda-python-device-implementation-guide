import os
import shutil
import sys

static_url="cuda-python-device-implementation-guide"

def update_references():
    # Update HTML references
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.html'):
                html_path = os.path.join(root, file)
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                new_content = content.replace('href="_static/', f'href="{static_url}/_static/')
                new_content = new_content.replace('src="_static/', f'src="{static_url}/_static/')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated static references in {html_path}")

if __name__ == "__main__":
    update_references()