#!/usr/bin/env python3
import os
import sys
import importlib.util
import argparse

def main():
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description='Run EduDPO scripts')
    parser.add_argument('script', help='Script name to run (without .py)')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments to pass to the script')
    
    args = parser.parse_args()
    
    # Construct the full path to the script
    script_path = os.path.join(project_root, 'scripts', f'{args.script}.py')
    
    if not os.path.exists(script_path):
        print(f"Error: Script {args.script}.py not found in scripts directory")
        sys.exit(1)
    
    # Load and run the script
    spec = importlib.util.spec_from_file_location(args.script, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Pass remaining arguments to the script's main function
    if hasattr(module, 'main'):
        sys.argv = [script_path] + args.args
        module.main()
    else:
        print(f"Error: Script {args.script}.py does not have a main() function")
        sys.exit(1)

if __name__ == '__main__':
    main() 