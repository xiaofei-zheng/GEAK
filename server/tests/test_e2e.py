#!/usr/bin/env python3
"""End-to-End Tests for GEAK Online Service.

Tests the complete workflow with test_silu and test_rocprim cases.
"""

import httpx
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# 测试配置
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("TEST_API_KEY")

# 读取测试文件
GEAK_V3_DIR = Path(__file__).parent.parent.parent / "geak_v3"


def load_file(path: str) -> str:
    """Load file content."""
    file_path = GEAK_V3_DIR / path
    if file_path.exists():
        return file_path.read_text()
    raise FileNotFoundError(f"File not found: {file_path}")


def get_headers():
    """Get request headers with API key."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def test_silu_case():
    """Test SiLU kernel optimization case (single file input)."""
    print("\n" + "=" * 60)
    print("Test Case 1: SiLU Kernel Optimization (Single File)")
    print("=" * 60)
    
    # Load test data
    try:
        silu_code = load_file("test_silu/silu.hip")
        silu_prompt = load_file("test_prompts/prompt_silu.md")
    except FileNotFoundError as e:
        print(f"⚠️  Skipping: {e}")
        return None
    
    print(f"✓ Loaded silu.hip ({len(silu_code)} bytes)")
    print(f"✓ Loaded prompt_silu.md ({len(silu_prompt)} bytes)")
    
    # Create task with multiple files support
    payload = {
        "input_type": "file",
        "files": [
            {
                "filename": "silu.hip",
                "content": silu_code,
            },
        ],
        "prompt": silu_prompt,
        "config": {
            "model": {
                "model_class": "amd_llm",
                "model_name": "claude-opus-4.5",
            }
        },
        "runtime": {
            "gpu_count": 1,
            "timeout": 3600,
        }
    }
    
    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=30.0) as client:
        # 1. Create task
        print("\n1. Creating task...")
        response = client.post("/api/v1/tasks", json=payload)
        
        if response.status_code != 201:
            print(f"   ❌ Failed to create task: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        task = response.json()
        task_id = task["id"]
        print(f"   ✅ Task created: {task_id}")
        print(f"   Status: {task['status']}")
        
        # 2. Get task details
        print("\n2. Getting task details...")
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        task = response.json()
        print(f"   ✅ Task ID: {task['id']}")
        print(f"   Input Type: {task['input_type']}")
        print(f"   Input Path: {task.get('input_path', 'N/A')}")
        
        # 3. Check outputs (should be empty for pending task, but directory exists)
        print("\n3. Checking outputs...")
        response = client.get(f"/api/v1/tasks/{task_id}/outputs")
        assert response.status_code == 200
        outputs = response.json()
        print(f"   ✅ Output path: {outputs['output_path']}")
        print(f"   Files: {len(outputs['files'])} file(s)")
        if outputs.get('sftp_command'):
            print(f"   SFTP: {outputs['sftp_command']}")
        
        # 4. Verify input file was saved
        print("\n4. Verifying input file...")
        input_path = task.get('input_path')
        if input_path and os.path.exists(input_path):
            print(f"   ✅ Input file exists: {input_path}")
            with open(input_path, 'r') as f:
                content = f.read()
            print(f"   File size: {len(content)} bytes")
        else:
            print(f"   ⚠️  Input path not accessible: {input_path}")
        
        return task_id


def test_rocprim_case():
    """Test rocPRIM optimization case (repository input)."""
    print("\n" + "=" * 60)
    print("Test Case 2: rocPRIM Binary Search (Repository)")
    print("=" * 60)
    
    # Load prompt
    try:
        rocprim_prompt = load_file("test_prompts/rocprim_device_binay_search.md")
    except FileNotFoundError as e:
        print(f"⚠️  Skipping: {e}")
        return None
    
    print(f"✓ Loaded rocprim prompt ({len(rocprim_prompt)} bytes)")
    
    # Create task with repo input
    payload = {
        "input_type": "repo",
        "repo": {
            "url": "https://github.com/ROCm/rocPRIM.git",
            "branch": "develop",
        },
        "prompt": rocprim_prompt,
        "runtime": {
            "gpu_count": 1,
            "timeout": 7200,
        }
    }
    
    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=120.0) as client:
        # 1. Create task (this will clone the repo, may take time)
        print("\n1. Creating task (cloning repository, may take a while)...")
        response = client.post("/api/v1/tasks", json=payload)
        
        if response.status_code != 201:
            print(f"   ❌ Failed to create task: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return None
        
        task = response.json()
        task_id = task["id"]
        print(f"   ✅ Task created: {task_id}")
        print(f"   Status: {task['status']}")
        
        # 2. Get task details
        print("\n2. Getting task details...")
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        task = response.json()
        print(f"   ✅ Task ID: {task['id']}")
        print(f"   Input Type: {task['input_type']}")
        print(f"   Input Path: {task.get('input_path', 'N/A')}")
        
        # 3. Check outputs
        print("\n3. Checking outputs...")
        response = client.get(f"/api/v1/tasks/{task_id}/outputs")
        assert response.status_code == 200
        outputs = response.json()
        print(f"   ✅ Output path: {outputs['output_path']}")
        
        return task_id


def test_list_and_filter():
    """Test listing and filtering tasks."""
    print("\n" + "=" * 60)
    print("Test Case 3: List and Filter Tasks")
    print("=" * 60)
    
    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=30.0) as client:
        # List all tasks
        print("\n1. Listing all tasks...")
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200
        data = response.json()
        print(f"   ✅ Total tasks: {data['total']}")
        
        for task in data['tasks'][:5]:  # Show first 5
            print(f"   - {task['id'][:8]}... | {task['status']} | {task['input_type']}")
        
        # Filter by status
        print("\n2. Filtering by status='pending'...")
        response = client.get("/api/v1/tasks", params={"status": "pending"})
        assert response.status_code == 200
        data = response.json()
        print(f"   ✅ Pending tasks: {len(data['tasks'])}")


def test_submit_task(task_id: str):
    """Test submitting a task for execution (optional)."""
    print("\n" + "=" * 60)
    print(f"Test Case 4: Submit Task for Execution")
    print("=" * 60)
    print(f"Task ID: {task_id}")
    
    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=60.0) as client:
        print("\n1. Submitting task to SaFE platform...")
        response = client.post(f"/api/v1/tasks/{task_id}/submit")
        
        if response.status_code == 200:
            task = response.json()
            print(f"   ✅ Task submitted successfully!")
            print(f"   Status: {task['status']}")
            print(f"   Workload ID: {task.get('safe_workload_id', 'N/A')}")
            return True
        else:
            print(f"   ❌ Failed to submit: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False


def test_download_file(task_id: str, file_path: str = "execution.log"):
    """Test downloading a file from task outputs."""
    print("\n" + "=" * 60)
    print(f"Test Case 5: Download File")
    print("=" * 60)
    
    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=30.0) as client:
        print(f"\n1. Downloading {file_path}...")
        response = client.get(
            f"/api/v1/tasks/{task_id}/download",
            params={"path": file_path}
        )
        
        if response.status_code == 200:
            print(f"   ✅ Downloaded successfully!")
            print(f"   Size: {len(response.content)} bytes")
            print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")
            return True
        elif response.status_code == 404:
            print(f"   ⚠️  File not found (task may not have run yet)")
            return False
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False


def run_all_tests():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("GEAK Online Service - End-to-End Tests")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY[:10]}...")
    
    # Test 1: SiLU case
    silu_task_id = test_silu_case()
    
    # Test 2: rocPRIM case (skip if too slow)
    # rocprim_task_id = test_rocprim_case()
    print("\n⚠️  Skipping rocPRIM test (repo clone takes too long)")
    
    # Test 3: List and filter
    test_list_and_filter()
    
    # Test 4: Submit task (optional - requires SaFE workload)
    if silu_task_id:
        print("\n" + "-" * 60)
        submit = input("Do you want to submit the task to SaFE platform? (y/N): ")
        if submit.lower() == 'y':
            test_submit_task(silu_task_id)
    
    # Test 5: Download (will fail for pending tasks)
    if silu_task_id:
        test_download_file(silu_task_id, "execution.log")
    
    print("\n" + "=" * 60)
    print("End-to-End Tests Completed!")
    print("=" * 60)


def run_quick_tests():
    """Run quick tests without interactive prompts."""
    print("=" * 60)
    print("GEAK Online Service - Quick E2E Tests")
    print("=" * 60)
    
    # Test 1: SiLU case - create task
    silu_task_id = test_silu_case()
    
    # Test 2: List and filter
    test_list_and_filter()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if silu_task_id:
        print(f"✅ SiLU task created: {silu_task_id}")
        print("\nTo submit this task for execution:")
        print(f"  curl -X POST {BASE_URL}/api/v1/tasks/{silu_task_id}/submit \\")
        print(f"       -H 'Authorization: Bearer {API_KEY}'")
    
    return silu_task_id is not None


def run_full_tests():
    """Run full end-to-end tests including submit, monitor, and download."""
    print("=" * 60)
    print("GEAK Online Service - Full E2E Tests")
    print("=" * 60)
    
    # Test 1: Create SiLU task
    silu_task_id = test_silu_case()
    if not silu_task_id:
        print("❌ Failed to create task")
        return False
    
    # Test 2: Submit task
    if not test_submit_task(silu_task_id):
        print("❌ Failed to submit task")
        return False
    
    # Test 3: Monitor execution
    print("\n" + "=" * 60)
    print("Monitoring Task Execution")
    print("=" * 60)
    
    output_dir = f"/wekafs/geak/tasks/1b028c9fa3819bb971f80bc74a90e21d/{silu_task_id}/output"
    max_wait = 300  # 5 minutes max
    wait_interval = 10
    
    for i in range(max_wait // wait_interval):
        time.sleep(wait_interval)
        
        # Check if execution.log exists and task completed
        log_path = f"{output_dir}/execution.log"
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                content = f.read()
            if "completed successfully" in content:
                print(f"   ✅ Task completed after {(i+1) * wait_interval} seconds")
                break
            # Show current step
            import re
            steps = re.findall(r'step (\d+)', content)
            if steps:
                print(f"   [{(i+1) * wait_interval}s] Executing... (step {steps[-1]})")
        else:
            print(f"   [{(i+1) * wait_interval}s] Waiting for execution to start...")
    else:
        print(f"   ⚠️ Timeout after {max_wait} seconds")
    
    # Test 4: Check outputs
    print("\n" + "=" * 60)
    print("Checking Outputs")
    print("=" * 60)
    
    with httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=30.0) as client:
        response = client.get(f"/api/v1/tasks/{silu_task_id}/outputs")
        if response.status_code == 200:
            outputs = response.json()
            print(f"   ✅ Output path: {outputs['output_path']}")
            print(f"   Files: {len(outputs['files'])}")
            for f in outputs['files']:
                print(f"     - {f['path']} ({f['size']} bytes)")
        else:
            print(f"   ❌ Failed to get outputs: {response.status_code}")
    
    # Test 5: Download files
    test_download_file(silu_task_id, "execution.log")
    test_download_file(silu_task_id, "silu.hip")
    
    # Test 6: List and filter
    test_list_and_filter()
    
    print("\n" + "=" * 60)
    print("Full E2E Tests Completed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_tests()
        sys.exit(0 if success else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "--full":
        success = run_full_tests()
        sys.exit(0 if success else 1)
    else:
        run_all_tests()
