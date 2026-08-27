from __future__ import annotations

import sys

from scripts.run_unified_preset import main

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/suites/w2_prf_core.json"])
    if "--split" not in sys.argv:
        sys.argv.extend(["--split", "configs/splits/adaptive_v1.json"])
    main()
