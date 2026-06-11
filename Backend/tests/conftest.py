import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
from app.main import app
from app.auth import get_current_user, get_current_officer, get_current_taxpayer
from types import SimpleNamespace

class PersistentOverridesDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistent_keys = {}

    def set_persistent(self, key, value):
        self.persistent_keys[key] = value
        self[key] = value

    def clear(self):
        super().clear()
        for k, v in self.persistent_keys.items():
            self[k] = v

    def __delitem__(self, key):
        if key in self.persistent_keys:
            del self.persistent_keys[key]
        super().__delitem__(key)

class MockUser:
    id = 1
    badge_id = "test-officer-001"
    role = "admin"
    username = "test_officer"
    email = "test_officer@gdt.gov.vn"

def mock_get_current_user():
    return MockUser()

def mock_get_current_officer():
    return MockUser()

def mock_get_current_taxpayer():
    return MockUser()

# Initialize the persistent dictionary on app.dependency_overrides
overrides = PersistentOverridesDict(app.dependency_overrides)
app.dependency_overrides = overrides

@pytest.fixture(autouse=True)
def setup_auth_overrides():
    # Setup mock overrides before each test
    overrides.set_persistent(get_current_user, mock_get_current_user)
    overrides.set_persistent(get_current_officer, mock_get_current_officer)
    overrides.set_persistent(get_current_taxpayer, mock_get_current_taxpayer)
    yield
