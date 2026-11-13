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
