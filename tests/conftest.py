"""Fixtures comunes: los tests corren sin SLURM ni GPU.

- La raíz del repo se agrega a sys.path para importar los scripts como módulos.
- Cada test corre en un directorio temporal con SLURM_JOB_ID falso.
- forgethreads, forgeffects y cupy se reemplazan por mocks en sys.modules.
"""
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAKE_JOB_ID = "4242"


@pytest.fixture(autouse=True)
def slurm_workdir(tmp_path, monkeypatch):
    """Directorio de trabajo temporal con logs/ y jobs_results/, y SLURM_JOB_ID falso."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", FAKE_JOB_ID)
    (tmp_path / "logs").mkdir()
    (tmp_path / "jobs_results").mkdir()
    return tmp_path


def _fake_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def fake_forgethreads(monkeypatch):
    """forgethreads falso: maxmin devuelve (None, None, eff_order); configurable vía .maxmin."""
    return _fake_module(
        monkeypatch, "forgethreads",
        maxmin=mock.Mock(return_value=(None, None, 3)),
        set_verbose=mock.Mock(),
    )


@pytest.fixture
def fake_cupy(monkeypatch):
    """cupy falso: get_default_memory_pool().free_all_blocks() registra las llamadas."""
    pool = mock.Mock()
    module = _fake_module(monkeypatch, "cupy", get_default_memory_pool=mock.Mock(return_value=pool))
    module.pool = pool
    return module


@pytest.fixture
def fake_forgeffects(monkeypatch):
    """forgeffects falso: FE configurable vía .FE (por defecto devuelve una lista vacía)."""
    return _fake_module(monkeypatch, "forgeffects", FE=mock.Mock(return_value=[]))
