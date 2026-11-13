import subprocess
import sys


def _loaded_after(statement: str, *names: str) -> dict[str, bool]:
    code = f"import sys\n{statement}\nprint(','.join(str(n in sys.modules) for n in {names!r}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    return dict(zip(names, (flag == "True" for flag in out.split(",")), strict=True))


def test_the_packer_does_not_pull_in_the_3mf_module():
    assert _loaded_after("import platenest", "platenest.bambu3mf") == {"platenest.bambu3mf": False}


def test_the_3mf_module_is_stdlib_only():
    assert _loaded_after("import platenest.bambu3mf", "cv2", "numpy") == {"cv2": False, "numpy": False}


def _runs(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_pack_is_the_function_whatever_was_imported_first():
    # 0.1.0: the function shared its name with the submodule defining it, so importing Cancelled or Progress
    # first made ``from platenest import pack`` hand back the module.
    _runs("from platenest import Cancelled, pack; assert callable(pack), pack")
    _runs("from platenest import Progress; from platenest import pack; assert callable(pack), pack")
    _runs("import platenest; platenest.Cancelled; assert callable(platenest.pack), platenest.pack")
    _runs("from platenest import pack, Cancelled, Progress, render_plate, Zone; assert callable(pack)")


def test_no_exported_name_is_also_a_submodule():
    import pkgutil

    import platenest

    submodules = {module.name for module in pkgutil.iter_modules(platenest.__path__)}
    assert not submodules & set(platenest.__all__), submodules & set(platenest.__all__)
