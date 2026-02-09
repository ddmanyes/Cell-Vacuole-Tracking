# Test Log

Use this log to record parameter trials and outcomes.

## Template

- Date:
- Data:
- Script:
- Frame(s):
- Method:
- Params:
- Output:
- Notes:

## Entries

### 2026-02-09
- Date: 2026-02-09
- Data: data/bafA1/Group 1_wellA1_RI_MIP_stitched.tiff
- Script: src/tests/bubble_frame_preproc_test.py
- Frame(s): 0
- Method: rb + CLAHE + threshold (within cellpose mask)
- Params: BUBBLE_TH_THRESH=0.28, BUBBLE_TH_MIN_AREA=20, BUBBLE_TH_MAX_AREA=None, BUBBLE_TH_MIN_CIRCULARITY=0.1
- Output: results/bubble_preproc_frame/frame_0_compare.png
- Notes: Selected "rb+clahe" for bubble stats.
