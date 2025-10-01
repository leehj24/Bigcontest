import os

def print_folder_structure(parent_folder, indent=""):
    try:
        items = os.listdir(parent_folder)
    except PermissionError:
        print(indent + "[권한 없음]")
        return

    for item in items:
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}/")
            print_folder_structure(item_path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

# 현재 작업 디렉토리 기준
current_dir = os.getcwd()
print(f"폴더 구조: {current_dir}")
print_folder_structure(current_dir)
