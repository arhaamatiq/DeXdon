"""Check that tendon ERROR diagnostics printed from a kernel inside a captured CUDA graph are caught and
attributed to the right world (the harness maps tendon index // 3 to a world).

    python experiments/check_error_capture.py
"""

import re
import sys
from pathlib import Path

import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexdon.finger import capture_tendon_errors

wp.config.quiet = True


@wp.kernel
def fake_error(ids: wp.array[int]):
    wp.printf("ERROR: Tendon %d material rest-length floor added %g m of cable. (capture test)\n", ids[wp.tid()], 0.02)


ids = wp.array([7, 4000], dtype=int)
errors: list[str] = []
with capture_tendon_errors(errors):
    if ids.device.is_cuda:
        with wp.ScopedCapture() as cap:
            wp.launch(fake_error, dim=2, inputs=[ids])
        wp.capture_launch(cap.graph)  # the capture pass prints once, the replay prints again
    else:
        wp.launch(fake_error, dim=2, inputs=[ids])
worlds = sorted({int(re.search(r"Tendon (\d+)", ln).group(1)) // 3 for ln in errors})
print(f"device {ids.device}, graph replay {ids.device.is_cuda}: captured {len(errors)} ERROR lines, worlds flagged {worlds}")
assert worlds == [2, 1333], worlds
print("OK: diagnostics printed inside the captured step are logged and mapped to worlds")
