from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteAuthConfig:
    host: str
    mode: str  # key | password
    ssh_key_path: str | None = None
    password: str | None = None

    def ssh_prefix(self) -> list[str]:
        base: list[str] = ["ssh"]
        if self.mode == "password":
            if not self.password:
                raise RuntimeError("password mode selected but REMOTE_PASSWORD is empty")
            if shutil.which("sshpass") is None:
                raise RuntimeError("password mode requires sshpass, but it is not installed")
            base = ["sshpass", "-p", self.password, *base]
        ssh_cmd = [*base, "-o", "StrictHostKeyChecking=no"]
        if self.ssh_key_path:
            ssh_cmd.extend(["-i", self.ssh_key_path])
        ssh_cmd.append(self.host)
        return ssh_cmd

    def scp_prefix(self) -> list[str]:
        base: list[str] = ["scp", "-q", "-o", "StrictHostKeyChecking=no"]
        if self.mode == "password":
            if not self.password:
                raise RuntimeError("password mode selected but REMOTE_PASSWORD is empty")
            if shutil.which("sshpass") is None:
                raise RuntimeError("password mode requires sshpass, but it is not installed")
            base = ["sshpass", "-p", self.password, *base]
        if self.ssh_key_path:
            base.extend(["-i", self.ssh_key_path])
        return base


def resolve_auth_config(
    *,
    host: str,
    auth_mode: str = "auto",
    ssh_key_path: str | None = None,
    remote_password: str | None = None,
) -> RemoteAuthConfig:
    mode = (auth_mode or "auto").strip().lower()
    password = remote_password if remote_password is not None else os.environ.get("REMOTE_PASSWORD", "")
    key = ssh_key_path or os.environ.get("REMOTE_SSH_KEY") or None

    if key:
        expanded = str(Path(key).expanduser())
        key = expanded

    if mode == "auto":
        if key:
            return RemoteAuthConfig(host=host, mode="key", ssh_key_path=key)
        if password:
            return RemoteAuthConfig(host=host, mode="password", password=password)
        return RemoteAuthConfig(host=host, mode="key", ssh_key_path=None)

    if mode == "key":
        return RemoteAuthConfig(host=host, mode="key", ssh_key_path=key)

    if mode == "password":
        return RemoteAuthConfig(host=host, mode="password", ssh_key_path=key, password=password)

    raise ValueError(f"unsupported auth_mode={auth_mode!r}; expected auto|key|password")
