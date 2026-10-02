"""Offline unit tests for login + token verification."""

import time

import jwt
import pytest

from medibot import auth
from medibot.config import JWT_SECRET


def test_good_credentials_give_role():
    assert auth.authenticate("nurse.priya", "nurse123") == "nurse"


@pytest.mark.parametrize("user,pw", [("nurse.priya", "wrong"), ("nobody", "x"), ("admin.sys", "")])
def test_bad_credentials_rejected(user, pw):
    with pytest.raises(auth.AuthError):
        auth.authenticate(user, pw)


def test_token_round_trip():
    claims = auth.verify_token(auth.issue_token("dr.mehta", "doctor"))
    assert claims["sub"] == "dr.mehta" and claims["role"] == "doctor"


def test_tampered_role_is_rejected():
    forged = jwt.encode({"sub": "nurse.priya", "role": "admin", "exp": time.time() + 60}, "not-the-secret", algorithm="HS256")
    with pytest.raises(auth.AuthError):
        auth.verify_token(forged)


def test_expired_token_is_rejected():
    old = jwt.encode({"sub": "x", "role": "nurse", "exp": 1}, JWT_SECRET, algorithm="HS256")
    with pytest.raises(auth.AuthError):
        auth.verify_token(old)
