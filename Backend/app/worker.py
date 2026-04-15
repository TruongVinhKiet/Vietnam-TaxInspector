"""
worker.py – Celery Worker Configuration
==========================================
Kết nối Celery với Redis để xử lý batch CSV ngầm (Background Task).

Sử dụng:
    celery -A app.worker worker --loglevel=info --pool=solo

Lưu ý Windows:
    - Cần thêm --pool=solo vì Celery trên Windows không hỗ trợ prefork
    - Nếu không có Redis, hệ thống tự fallback sang xử lý đồng bộ
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery

    celery_app = Celery(
        "taxinspector",
        broker=REDIS_URL,
        backend=REDIS_URL,
        include=["app.tasks"],
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="Asia/Ho_Chi_Minh",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_max_tasks_per_child=50,
        result_expires=3600,
    )

    CELERY_AVAILABLE = True
    print("[OK] Celery worker configured with Redis broker")

except ImportError:
    celery_app = None
    CELERY_AVAILABLE = False
    # Đã ẩn cảnh báo Celery vì hệ thống tự động fallback an toàn bằng luồng ngầm (threading)
    # print("[WARN] Celery not installed – batch processing will run synchronously")
