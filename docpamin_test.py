import requests
import os
import time
import json

headers = {"Authorization": "bearer 4mIxVAz8vg8irm3eqD1ZrTttZQKue67NjVhE56QHdCM"}
BASE_URL = "https://docpamin.superaip.samsungds.net/api/v1"
CRT_FILE = "/etc/ssl/certs/ca-certificates.crt"

task_id = None

# Task 생성
files = {"file": open("/mnt/data/kaiadmin/2503.09516v5.pdf", "rb")}
data  = {
    "alarm_options":    json.dumps({"enabled": True, "method": "email"}),
    "workflow_options": json.dumps({
        "workflow": "dp-o1",
        "worker_options": {
            "docling_to_formats": ["md", "json"],
            "docling_image_export_mode": "embedded",
        },
    }),
}
resp = requests.post(f"{BASE_URL}/tasks", headers=headers, files=files, data=data, verify=CRT_FILE)
task_id = resp.json().get("task_id")
print(task_id)


# Task 조회
while True:
    resp = requests.get(f"{BASE_URL}/tasks/{task_id}", headers=headers, verify=CRT_FILE)
    status = resp.json().get("status")
    print(status)
    if status == 'DONE':
        file_ids = resp.json().get("files")
        print("DONE")
        break
    time.sleep(3)


# Task 결과 다운로드
options = {
  "task_ids": [task_id],
  "output_types": ["markdown", "json"]
}

resp = requests.post(f"{BASE_URL}/tasks/export", headers=headers, json=options, verify=CRT_FILE)

zip_filename = "file_name_to_save.zip"  # 저장할 파일 이름
with open(zip_filename, "wb") as f:
    f.write(resp.content)

