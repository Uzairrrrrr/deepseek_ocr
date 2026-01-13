#!/usr/bin/env python3
"""
Fix DeepSeek-OCR for CPU-only usage
Based on: https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/54

This script patches the cached DeepSeek-OCR model to work on CPU
by replacing .cuda() calls with .to(self.device)
"""

import os
import re
import sys

def find_model_file():
    """Find the cached model file"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR")
    
    print(f"🔍 Searching for model in: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found!")
        print(f"   The model hasn't been downloaded yet.")
        print(f"   Run the server once first to download it.")
        return None
    
    # Find modeling_deepseekocr.py
    for root, dirs, files in os.walk(cache_dir):
        if "modeling_deepseekocr.py" in files:
            return os.path.join(root, "modeling_deepseekocr.py")
    
    return None


def patch_model_file(model_file):
    """Patch the model file for CPU usage"""
    
    print(f"✓ Found model file: {model_file}")
    
    # Read original content
    with open(model_file, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Create backup
    backup_file = model_file + ".backup"
    if not os.path.exists(backup_file):
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"✓ Created backup: {backup_file}")
    else:
        print(f"ℹ️  Backup already exists: {backup_file}")
    
    # Apply patches
    content = original_content
    changes = []
    
    # Fix 1: Replace .cuda() with .to(self.device)
    if '.cuda()' in content:
        before_count = content.count('.cuda()')
        content = content.replace('.cuda()', '.to(self.device)')
        changes.append(f"Replaced {before_count} .cuda() calls")
    
    # Fix 2: Replace .to('cuda') with .to(self.device)
    patterns = [
        (r"\.to\(['\"]cuda['\"]\)", ".to(self.device)"),
        (r"\.to\(device=['\"]cuda['\"]\)", ".to(device=self.device)"),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append(f"Fixed hardcoded cuda strings")
    
    # Check if changes were made
    if content == original_content:
        print("ℹ️  No changes needed - file may already be patched")
        return False
    
    # Save patched content
    with open(model_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Successfully patched model file!")
    for change in changes:
        print(f"   • {change}")
    
    return True


def main():
    print("=" * 70)
    print("  DeepSeek-OCR CPU Patcher")
    print("=" * 70)
    print()
    
    # Find model file
    model_file = find_model_file()
    
    if not model_file:
        print("\n❌ Model file not found!")
        print("\n📝 To fix this:")
        print("   1. Run the server once to download the model:")
        print("      python3 app_enhanced.py")
        print("   2. Wait for it to fail (this downloads the model)")
        print("   3. Run this script again")
        print("   4. Then restart the server")
        sys.exit(1)
    
    # Patch the file
    patched = patch_model_file(model_file)
    
    if patched:
        print("\n🎉 Model successfully patched for CPU usage!")
        print("\n📝 Next steps:")
        print("   1. Restart your server:")
        print("      python3 app_enhanced.py")
        print("   2. Test with demo:")
        print("      python3 demo.py")
    else:
        print("\nℹ️  Model already appears to be patched")
    
    print("\n💡 To restore original:")
    print(f"   cp {model_file}.backup {model_file}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()