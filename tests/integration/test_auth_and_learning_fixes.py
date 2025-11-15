"""
Integration tests for Phase 1 & 2 Bug Fixes

Tests:
- Phase 1: API typo fix (feedback_request.apply_immediately)
- Phase 2: Database-backed user authentication

These tests validate that both critical bugs have been fixed.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import os
import sys
from datetime import datetime

# Add root directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.app import app, get_db
from src.database.models import Base, User as UserModel

# Use an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Create tables
Base.metadata.create_all(bind=test_engine)

def override_get_db():
    """Override get_db dependency for tests"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Apply the override BEFORE creating TestClient
app.dependency_overrides[get_db] = override_get_db

# Disable startup/shutdown events for testing (no Redis/engine needed)
app.router.on_startup.clear()
app.router.on_shutdown.clear()

# Create client AFTER override is set
client = TestClient(app, raise_server_exceptions=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Mock engine for learning feedback tests
from unittest.mock import Mock
import api.app as app_module

# Create a minimal mock engine with feedback_manager
mock_engine = Mock()
mock_feedback_manager = Mock()
mock_feedback_manager.add_feedback.return_value = "test_pattern_id_12345678"
mock_feedback_manager.approve_pattern.return_value = True
mock_engine.feedback_manager = mock_feedback_manager
app_module.engine = mock_engine

@pytest.fixture(scope="module")
def test_db_user():
    """Fixture to create a persistent test user in the DB"""
    db = TestingSessionLocal()

    # Hashed password for 'admin123'
    hashed_password = pwd_context.hash("admin123")

    db_user = UserModel(
        username="db_admin",
        full_name="Database Admin",
        email="db_admin@hospital.org",
        hashed_password=hashed_password,
        role="admin",
        permissions=["read", "write", "approve"],
        is_active=True
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    yield db_user

    # Clean up
    db.delete(db_user)
    db.commit()
    db.close()

class TestAuthFixes:
    """Tests for Phase 2: Database-backed authentication"""

    def test_database_backed_login_success(self, test_db_user):
        """
        Test: Can log in as a user stored in the database?
        This validates Fix #2.
        """
        response = client.post(
            "/api/auth/login",
            data={"username": "db_admin", "password": "admin123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user_info"]["username"] == "db_admin"
        assert data["user_info"]["role"] == "admin"

    def test_database_backed_login_failure(self):
        """Test: Fails to log in as a non-existent user"""
        response = client.post(
            "/api/auth/login",
            data={"username": "ghost", "password": "password"}
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_get_current_user_from_db(self, test_db_user):
        """Test: /api/auth/me endpoint correctly fetches user from DB"""
        # Log in to get token
        login_response = client.post(
            "/api/auth/login",
            data={"username": "db_admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]

        # Test /me endpoint
        me_response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert me_response.status_code == 200
        data = me_response.json()
        assert data["username"] == "db_admin"
        assert data["email"] == "db_admin@hospital.org"


class TestLearningApiFixes:
    """Tests for Phase 1: API Typo Fix"""

    def test_submit_feedback_with_auto_approve(self, test_db_user):
        """
        Test: Can an admin submit feedback with apply_immediately=True?
        This validates Fix #1.
        """
        # Log in as admin to get token
        login_response = client.post(
            "/api/auth/login",
            data={"username": "db_admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]

        feedback_data = {
            "uncertainty_id": "unc_test_123",
            "original_extraction": "POD#5",
            "correction": "Post-Operative Day 5",
            "context": {"fact_type": "temporal_reference"},
            "apply_immediately": True  # This triggers the fixed line
        }

        response = client.post(
            "/api/learning/feedback",
            headers={"Authorization": f"Bearer {token}"},
            json=feedback_data
        )

        # If the typo existed, this would 500. A 200 proves the fix.
        assert response.status_code == 200
        data = response.json()
        assert data["pattern_status"] == "APPROVED"
        assert "auto-approved" in data["message"]
