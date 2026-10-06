import json

import numpy as np
import pytest

import fe_job
from parser import get_fe_parser


def make_input(path, k=1, m=3, n=2, meta=None, arrays=None, skip=()):
    """Crea un directorio de entrada válido para fe_job; meta/arrays sobrescriben partes."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = {
        "CC": rng.uniform(size=(k, m, m)).astype(np.float32),
        "CE": rng.uniform(size=(k, m, n)).astype(np.float32),
        "EE": rng.uniform(size=(k, n, n)).astype(np.float32),
    }
    data.update(arrays or {})
    for name, array in data.items():
        if name not in skip:
            np.save(path / f"{name}.npy", array)
    content = {"causes": [f"c{i}" for i in range(m)], "effects": [f"e{j}" for j in range(n)],
               "thr": 0.5, "maxorder": 3, "reps": 1}
    content.update(meta or {})
    if "meta.json" not in skip:
        (path / "meta.json").write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def errors_of(path):
    with pytest.raises(fe_job.InputError) as exc:
        fe_job.load_and_validate(path)
    return str(exc.value)


# --- parser -----------------------------------------------------------------

def test_parser_requires_input_dir():
    assert get_fe_parser().parse_args(["--input-dir", "jobs_inputs/abc"]).input_dir == "jobs_inputs/abc"
    with pytest.raises(SystemExit):
        get_fe_parser().parse_args([])


# --- entradas válidas ---------------------------------------------------------

@pytest.mark.parametrize("k", [1, 3])
def test_valid_input_loads_as_float32(tmp_path, k):
    data = fe_job.load_and_validate(make_input(tmp_path / "in", k=k, meta={"seed": 7}))
    assert data["CC"].shape == (k, 3, 3) and data["CE"].shape == (k, 3, 2) and data["EE"].shape == (k, 2, 2)
    assert all(data[x].dtype == np.float32 for x in ("CC", "CE", "EE"))
    assert data["causes"] == ["c0", "c1", "c2"] and data["effects"] == ["e0", "e1"]
    assert (data["thr"], data["maxorder"], data["reps"], data["seed"]) == (0.5, 3, 1, 7)


def test_seed_is_optional(tmp_path):
    assert fe_job.load_and_validate(make_input(tmp_path / "in"))["seed"] is None


def test_float64_and_border_values_are_accepted(tmp_path):
    cc = np.zeros((1, 3, 3))
    cc[0, 0, 1] = 1.0
    data = fe_job.load_and_validate(make_input(tmp_path / "in", arrays={"CC": cc}, meta={"thr": 1}))
    assert data["CC"].dtype == np.float32 and data["thr"] == 1.0


# --- errores ----------------------------------------------------------------

def test_missing_dir(tmp_path):
    assert "no existe" in errors_of(tmp_path / "nada")


@pytest.mark.parametrize("missing", ["CC", "CE", "EE", "meta.json"])
def test_missing_file(tmp_path, missing):
    assert missing in errors_of(make_input(tmp_path / "in", skip=(missing,)))


def test_invalid_meta_json(tmp_path):
    path = make_input(tmp_path / "in", skip=("meta.json",))
    (path / "meta.json").write_text("{no es json")
    assert "meta.json" in errors_of(path)


@pytest.mark.parametrize("arrays, expected", [
    ({"CC": np.zeros((1, 3, 4), np.float32)}, "CC"),            # no cuadrada
    ({"CE": np.zeros((1, 3, 5), np.float32)}, "CE"),            # n distinto de EE
    ({"EE": np.zeros((2, 2, 2), np.float32)}, "k"),             # k distinto
    ({"CC": np.zeros((3, 3), np.float32)}, "3D"),               # no 3D
])
def test_inconsistent_shapes(tmp_path, arrays, expected):
    assert expected in errors_of(make_input(tmp_path / "in", arrays=arrays))


@pytest.mark.parametrize("bad", [1.5, -0.1, np.nan, np.inf])
def test_values_outside_unit_interval(tmp_path, bad):
    ce = np.full((1, 3, 2), 0.5, np.float32)
    ce[0, 1, 1] = bad
    assert "[0,1]" in errors_of(make_input(tmp_path / "in", arrays={"CE": ce}))


def test_non_numeric_array(tmp_path):
    assert "numérico" in errors_of(make_input(tmp_path / "in", arrays={"EE": np.full((1, 2, 2), True)}))


@pytest.mark.parametrize("meta, expected", [
    ({"causes": ["a", "b"]}, "causes"),                 # largo distinto de m
    ({"effects": ["x", "y", "z"]}, "effects"),
    ({"causes": ["a", "b", 3]}, "causes"),              # no todos str
    ({"effects": ["c0", "e1"]}, "repetidas"),           # etiqueta duplicada entre causes y effects
    ({"thr": 1.5}, "thr"),
    ({"thr": "0.5"}, "thr"),
    ({"thr": True}, "thr"),
    ({"maxorder": 1}, "maxorder"),
    ({"maxorder": 2.5}, "maxorder"),
    ({"reps": 0}, "reps"),
    ({"seed": "x"}, "seed"),
])
def test_invalid_meta_fields(tmp_path, meta, expected):
    assert expected in errors_of(make_input(tmp_path / "in", meta=meta))


@pytest.mark.parametrize("field", ["causes", "effects", "thr", "maxorder", "reps"])
def test_missing_meta_field(tmp_path, field):
    path = make_input(tmp_path / "in")
    meta = json.loads((path / "meta.json").read_text())
    del meta[field]
    (path / "meta.json").write_text(json.dumps(meta))
    assert field in errors_of(path)


def test_all_errors_reported_together(tmp_path):
    message = errors_of(make_input(tmp_path / "in", arrays={"CC": np.zeros((1, 3, 4), np.float32)},
                                   meta={"thr": 2, "reps": 0}))
    assert "CC" in message and "thr" in message and "reps" in message


def test_main_exits_nonzero_with_clear_message(tmp_path, capsys, fake_forgeffects):
    path = make_input(tmp_path / "in", meta={"maxorder": 1})
    with pytest.raises(SystemExit) as exc:
        fe_job.main(["--input-dir", str(path)])
    assert exc.value.code != 0
    assert "maxorder" in capsys.readouterr().err
    fake_forgeffects.FE.assert_not_called()
