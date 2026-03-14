import importlib.metadata
import os
import sys

def check_package(package_name, min_version):
    try:
        dist_name = package_name
        if package_name == 'pyyaml':
            dist_name = 'PyYAML'
        elif package_name == 'scikit-learn':
            dist_name = 'scikit-learn'
            
        version = importlib.metadata.version(dist_name)
        
        v1_parts = [int(p) for p in version.split('.')[:2] if p.isdigit()]
        v2_parts = [int(p) for p in min_version.split('.')[:2] if p.isdigit()]
        
        while len(v1_parts) < 2: v1_parts.append(0)
        while len(v2_parts) < 2: v2_parts.append(0)
            
        is_valid = True
        if v1_parts[0] < v2_parts[0]:
            is_valid = False
        elif v1_parts[0] == v2_parts[0] and v1_parts[1] < v2_parts[1]:
            is_valid = False

        if is_valid:
            print(f"[OK] {package_name} (found {version})")
            return True
        else:
            print(f"[FAIL] {package_name} (found {version}, requires >={min_version})")
            return False
            
    except importlib.metadata.PackageNotFoundError:
        print(f"[FAIL] {package_name} is not installed (requires >={min_version})")
        return False
    except Exception as e:
        print(f"[FAIL] {package_name}: Error checking version - {str(e)}")
        return False

def check_env_var(var_name):
    if os.environ.get(var_name):
        print(f"[OK] {var_name}")
        return True
    else:
        print(f"[FAIL] {var_name}")
        return False

def main():
    all_passed = True
    
    dependencies = {
        "torch": "2.1.0",
        "nni": "3.0",
        "openai": "1.10.0",
        "anthropic": "0.20.0",
        "tensorflow": "2.14.0",
        "pyyaml": "6.0",
        "tqdm": "4.66.0",
        "numpy": "1.24.0",
        "scikit-learn": "1.3.0",
        "rich": "13.0.0",
        "click": "8.1.0",
        "fastapi": "0.109.0",
        "uvicorn": "0.27.0"
    }

    for pkg, ver in dependencies.items():
        if not check_package(pkg, ver):
            all_passed = False

    if not check_env_var("OPENAI_API_KEY"):
        all_passed = False
    if not check_env_var("ANTHROPIC_API_KEY"):
        all_passed = False

    if all_passed:
        print("All checks passed.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
