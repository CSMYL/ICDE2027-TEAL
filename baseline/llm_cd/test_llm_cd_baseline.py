import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # CAIM_code/
sys.path.insert(0, project_root)

from baseline.llm_cd.llm_cd_baseline import main


if __name__ == "__main__":
    main()
